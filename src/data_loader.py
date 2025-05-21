import pandas as pd

from .config import BANK_FULL, BANK_ADD, RANDOM_STATE
from sklearn.model_selection import train_test_split

def load_preprocessed(prep_dir: str):
    X_train = pd.read_csv(f"{prep_dir}/X_train_preprocessed.csv")
    y_train = pd.read_csv(f"{prep_dir}/y_train.csv")['y']
    X_test  = pd.read_csv(f"{prep_dir}/X_test_preprocessed.csv")
    y_test  = pd.read_csv(f"{prep_dir}/y_test.csv")['y']
    return X_train, y_train, X_test, y_test


def load_raw():
    bank_full = pd.read_csv(BANK_FULL, sep=';')
    bank_add  = pd.read_csv(BANK_ADD,  sep=';')
    return bank_full, bank_add


def train_test_split_df(df, target_col='y', test_size=0.2):
    X = df.drop(columns=[target_col])
    y = df[target_col].map({'no':0,'yes':1})
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE
    )
