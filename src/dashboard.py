import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    BINARY_ORDER,
    BANK_ADD,
    BANK_FULL,
    DAY_ORDER,
    EDU_ORDER,
    MARITAL_ORDER,
    MODELS_DIR,
    MONTH_ORDER,
    POUTCOME_ORDER,
    PREP_DIR,
    TERNARY_ORDER,
    TUNED_PARAMS_PATH
)
from src.data_loader import (
    load_preprocessed,
    load_raw,
    train_test_split_df,
)
from src.eda import (
    correlation_matrix,
    numeric_distribution,
    ordered_category_freq,
    subscription_rate_by_category,
    unknown_placeholder_counts,
)
from src.evaluation import (
    evaluate_models,
    plot_conf_matrices,
    plot_pr,
    plot_roc,
    summary_table,
    threshold_analysis,
)
from src.modeling import (
    build_stacked,
    tune_all,
)
from src.preprocessing import (
    build_preprocessor,
    engineer_features,
)


st.set_page_config(
    page_title="Bank Marketing Term-Deposit Subscription Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Bank Marketing Term-Deposit Subscription Dashboard")

# --- Sidebar navigation ---
page = st.sidebar.selectbox(
    "Steps",
    ["Exploratory Data Analysis (EDA", "Feature Engineering and Preprocessing", "Model Tuning", "Model Evaluation"]
)

# --- Exploratory Data Analysis (EDA) ---
if page == "Exploratory Data Analysis (EDA":
    st.header("Exploratory Data Analysis (EDA)")

    # Load raw datasets
    bank_full, bank_add = load_raw()
    st.subheader("Dataset Shapes")
    st.write(f"- Raw (bank-full): {bank_full.shape[0]} rows × {bank_full.shape[1]} cols")
    st.write(f"- Enriched (bank-additional-full): {bank_add.shape[0]} rows × {bank_add.shape[1]} cols")
    st.write("The enriched data is chosen to be used.")

    # Unknown placeholder counts
    st.subheader("'unknown' Placeholder Counts")
    fig_unknown = unknown_placeholder_counts(bank_add)
    st.plotly_chart(fig_unknown, use_container_width=True)

    # Numeric distributions
    st.subheader("Numeric Feature Distributions")
    numeric_cols = bank_add.select_dtypes(include=[np.number]).columns.tolist()
    dist_figs = numeric_distribution(bank_add, numeric_cols)
    for col, fig in dist_figs.items():
        st.plotly_chart(fig, use_container_width=True)

    # Ordered categorical frequencies & subscription rate
    st.subheader("Categorical Frequencies & Subscription Rates")
    ordered_cols = [
        ("month", MONTH_ORDER),
        ("day_of_week", DAY_ORDER),
        ("education", EDU_ORDER),
        ("marital", MARITAL_ORDER),
        ("poutcome", POUTCOME_ORDER)
    ]
    for col_name, order in ordered_cols:
        freq_fig = ordered_category_freq(bank_add, col_name, order)
        rate_fig = subscription_rate_by_category(bank_add, col_name, order)
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.plotly_chart(freq_fig, use_container_width=True)
        
        with c2:
            st.plotly_chart(rate_fig, use_container_width=True)


    # Correlation matrix
    st.subheader("Correlation Matrix (Numerical + Target)")
    bank_add["y_binary"] = bank_add["y"].map({'no':0,'yes':1})
    corr_fig = correlation_matrix(bank_add, numeric_cols + ["y_binary"])
    st.plotly_chart(corr_fig, use_container_width=True)


# --- Feature Engineering and Preprocessing ---
elif page == "Feature Engineering and Preprocessing":
    st.header("Feature Engineering and Preprocessing")

    # Load enriched dataset and engineer new features
    bank = pd.read_csv(BANK_ADD, sep=';')
    bank_engineered = engineer_features(bank)

    st.subheader("Engineered Data Snapshot")
    st.write(f"Shape after engineering: {bank_engineered.shape}")
    st.dataframe(bank_engineered.head(), height=200)

    # Split into train/test
    st.subheader("Train/Test Split (Stratified)")
    X_train, X_test, y_train, y_test = train_test_split_df(bank_engineered, target_col="y")
    st.write(f"- Training set: {X_train.shape}, Positive rate: {y_train.mean():.2%}")
    st.write(f"- Test set: {X_test.shape}, Positive rate: {y_test.mean():.2%}")
    st.write("~11% positive (“yes”) subscription rate preserved across splits.")
    
    # Build and display the preprocessing pipeline
    st.subheader("Preprocessing Pipeline")
    numeric_feats = [
        'age','campaign','pdays','previous','emp.var.rate',
        'cons.price.idx','cons.conf.idx','euribor3m','nr.employed',
        'no_prev_contact','month_sin','month_cos','dow_sin','dow_cos'
    ]
    categorical_feats = ['job','marital','education','default','housing','loan','contact','poutcome']
    preprocessor = build_preprocessor(numeric_feats, categorical_feats)
    st.code(preprocessor, language="python")


# --- Model Tuning ---
elif page == "Model Tuning":
    st.header("Hyperparameter Tuning (5-Fold Stratified CV)")

    bank = pd.read_csv(BANK_ADD, sep=';')
    bank_engineered = engineer_features(bank)
    X_train, X_test, y_train, y_test = train_test_split_df(bank_engineered, target_col="y")

    numeric_feats = [
        'age','campaign','pdays','previous','emp.var.rate',
        'cons.price.idx','cons.conf.idx','euribor3m','nr.employed',
        'no_prev_contact','month_sin','month_cos','dow_sin','dow_cos'
    ]
    categorical_feats = ['job','marital','education','default','housing','loan','contact','poutcome']
    preprocessor = build_preprocessor(numeric_feats, categorical_feats)

    if st.button("▶️ Run Hyperparameter Tuning (this may take several minutes)"):
        with st.spinner("Tuning models..."):
            tuned = tune_all(X_train, y_train, preprocessor, use_saved=True)

        summary = pd.DataFrame({
            name: {
                "CV AUC": tuned[name]["cv_auc"],
                "Best Params": tuned[name]["best_params"]
            }
            for name in tuned
        }).T
        st.subheader("Tuning Results")
        st.dataframe(summary)

        st.subheader("Stacked Ensemble Pipeline")
        stacked_pipe = build_stacked(tuned, preprocessor)
        st.code(stacked_pipe, language="python")

        st.session_state["tuned_models"] = tuned
        st.session_state["stacked_pipe"] = stacked_pipe


# --- Model Evaluation ---
elif page == "Model Evaluation":
    st.header("Model Evaluation")

    if "tuned_models" not in st.session_state:
        st.error("Please run hyperparameter tuning on the 'Model Tuning' page first.")
        st.stop()

    tuned = st.session_state["tuned_models"]
    stacked_pipe = st.session_state["stacked_pipe"]
    X_train, y_train, X_test, y_test = load_preprocessed(PREP_DIR)

    models_for_eval = {name: tuned[name]["best_estimator"] for name in tuned}
    models_for_eval["Stacked"] = stacked_pipe

    results = evaluate_models(models_for_eval, X_test, y_test)

    # Metrics summary
    st.subheader("Test-Set Performance Summary")
    df_metrics = summary_table(results)
    st.dataframe(df_metrics)

    # ROC Curves
    st.subheader("ROC Curves")
    st.plotly_chart(plot_roc(results, y_test), use_container_width=True)

    # Precision–Recall Curves
    st.subheader("Precision–Recall Curves")
    st.plotly_chart(plot_pr(results, y_test), use_container_width=True)

    # Confusion Matrices
    st.subheader("Confusion Matrices")
    cms = plot_conf_matrices(results, y_test)
    for name, fig in cms.items():
        st.markdown(f"**{name}**")
        st.plotly_chart(fig, use_container_width=True)

    # Threshold Analysis for Stacked
    st.subheader("Threshold Analysis (Stacked Ensemble)")
    proba_stack = results["Stacked"]["Proba"]
    st.plotly_chart(threshold_analysis(proba_stack, y_test), use_container_width=True)
