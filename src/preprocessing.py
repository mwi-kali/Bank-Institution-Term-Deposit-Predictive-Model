import numpy as np
import pandas as pd

from .config import (
    MONTH_ORDER, DAY_ORDER, EDU_ORDER, MARITAL_ORDER,
    POUTCOME_ORDER, TERNARY_ORDER
)


from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from pandas.api.types import CategoricalDtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder




def build_full_pipeline(preprocessor, model):
    return ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', model)
    ])


def build_preprocessor(numeric_feats, categorical_feats):
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    return ColumnTransformer([
        ('num', num_pipe, numeric_feats),
        ('cat', cat_pipe, categorical_feats)
    ], remainder='drop')


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['no_prev_contact'] = (df['pdays']==999).astype(int)
    df['pdays'] = df['pdays'].replace(999, np.nan)
    cat_cols = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq<0.01].index
        df[col] = df[col].replace(rare, 'other')

    df['month'] = pd.Categorical(df['month'], categories=MONTH_ORDER, ordered=True)
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=DAY_ORDER, ordered=True)

    df['month_num'] = df['month'].cat.codes + 1
    df['dow_num'] = df['day_of_week'].cat.codes

    df['month_sin'] = np.sin(2*np.pi*df['month_num']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month_num']/12)
    df['dow_sin'] = np.sin(2*np.pi*df['dow_num']/5)
    df['dow_cos'] = np.cos(2*np.pi*df['dow_num']/5)

    return df

