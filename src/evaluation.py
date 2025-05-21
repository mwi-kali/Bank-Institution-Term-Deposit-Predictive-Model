import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, precision_recall_curve,
    confusion_matrix
)


def evaluate_models(models: dict, X_test, y_test):
    results = {}
    for name, pipe in models.items():
        proba = pipe.predict_proba(X_test)[:,1]
        pred  = (proba>=0.5).astype(int)
        results[name] = {
            'Accuracy':  accuracy_score(y_test, pred),
            'Precision': precision_score(y_test, pred),
            'Recall':    recall_score(y_test, pred),
            'F1 Score':  f1_score(y_test, pred),
            'ROC AUC':   auc(*roc_curve(y_test, proba)[:2]),
            'Proba':     proba
        }
    return results


def plot_conf_matrices(results, y_test):
    figs = {}
    for name, r in results.items():
        cm = confusion_matrix(y_test, (r['Proba']>=0.5).astype(int))
        figs[name] = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                             labels={'x':'Pred','y':'True'},
                             x=['No','Yes'], y=['No','Yes'],
                             title=f"{name} Confusion Matrix")
    return figs


def plot_pr(results, y_test):
    fig = go.Figure()
    for name, r in results.items():
        prec, rec, _ = precision_recall_curve(y_test, r['Proba'])
        ap = auc(rec, prec)
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode='lines',
            name=f"{name} (AP={ap:.3f})"
        ))
    fig.update_layout(title="Precision–Recall Curves", xaxis_title="Recall", yaxis_title="Precision")
    return fig


def plot_roc(results, y_test):
    fig = go.Figure()
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r['Proba'])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f"{name} (AUC={r['ROC AUC']:.3f})"
        ))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                             line=dict(dash='dash'),
                             name='Random'))
    fig.update_layout(title="ROC Curves",
                      xaxis_title="FPR", yaxis_title="TPR")
    return fig


def summary_table(results):
    df = pd.DataFrame({
        name:{
            'Accuracy':  r['Accuracy'],
            'Precision': r['Precision'],
            'Recall':    r['Recall'],
            'F1 Score':  r['F1 Score'],
            'ROC AUC':   r['ROC AUC']
        } for name,r in results.items()
    }).T.round(3)
    return df


def threshold_analysis(proba, y_test):
    prec, rec, thr = precision_recall_curve(y_test, proba)
    f1 = 2*prec*rec/(prec+rec+1e-8)
    df = pd.DataFrame({'threshold':thr,'precision':prec[:-1],'recall':rec[:-1],'f1':f1[:-1]})
    return px.line(df, x='threshold', y=['precision','recall','f1'],
                   labels={'value':'Metric','variable':'Metric'},
                   title="Precision, Recall & F1 vs Threshold")


