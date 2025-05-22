# src/modeling.py
import os
import json

from .config import RANDOM_STATE, TUNED_PARAMS_PATH

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier




def build_stacked(tuned, preprocessor):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    stack = StackingClassifier(
        estimators=[
            ('xgb', tuned['XGBoost']['best_estimator'].named_steps['clf']),
            ('mlp', tuned['MLP'   ]['best_estimator'].named_steps['clf'])
        ],
        final_estimator=LogisticRegression(random_state=RANDOM_STATE),
        passthrough=True,
        cv=cv,
        n_jobs=-1
    )

    pipe = ImbPipeline([
        ('prep',  preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('clf',   stack)
    ])
    return pipe


def get_model_specs():
    return {
        'LogisticRegression': {
            'estimator': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            'params': {'C': [0.01, 0.1, 1, 10]}
        },
        'RandomForest': {
            'estimator': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [100, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 10]
            }
        },
        'XGBoost': {
            'estimator': XGBClassifier(eval_metric='logloss',
                                       use_label_encoder=False,
                                       random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [100, 300],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            }
        },
        'SVM': {
            'estimator': SVC(probability=True, random_state=RANDOM_STATE),
            'params': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
        },
        'MLP': {
            'estimator': MLPClassifier(max_iter=1000, random_state=RANDOM_STATE),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [1e-4, 1e-3],
                'learning_rate_init': [1e-3, 1e-4]
            }
        }
    }


def load_tuned_params(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)
    

def tune_all(X, y, preprocessor, use_saved: bool=True):
    specs = get_model_specs()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    tuned = {}

    if use_saved and os.path.exists(TUNED_PARAMS_PATH):
        saved = load_tuned_params(TUNED_PARAMS_PATH)
        for name, spec in specs.items():
            params = saved[name]['best_params']
            pipe = ImbPipeline([
                ('prep',  preprocessor),
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('clf',   spec['estimator'])
            ])
            pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})
            pipe.fit(X, y)

            tuned[name] = {
                'best_estimator': pipe,
                'cv_auc':        saved[name]['cv_auc'],
                'best_params':   params
            }

    else:
        for name, spec in specs.items():
            pipe = ImbPipeline([
                ('prep',  preprocessor),
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('clf',   spec['estimator'])
            ])
            grid = GridSearchCV(
                pipe,
                {f"clf__{k}": v for k, v in spec['params'].items()},
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            print(f"Tuning {name}…")
            grid.fit(X, y)

            tuned[name] = {
                'best_estimator': grid.best_estimator_,
                'cv_auc':        grid.best_score_,
                'best_params':   {
                    k.replace('clf__', ''): v
                    for k, v in grid.best_params_.items()
                }
            }

        with open(TUNED_PARAMS_PATH, 'w') as f:
            json.dump(
                {name: {'cv_auc': tuned[name]['cv_auc'],
                        'best_params': tuned[name]['best_params']}
                 for name in tuned},
                f, indent=2
            )

    return tuned

