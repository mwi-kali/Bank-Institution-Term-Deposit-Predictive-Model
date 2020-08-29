import sys
import pydotplus

import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from numpy import std
from numpy import mean
from sklearn import svm
from scipy import stats
from io import StringIO
from keras import regularizers
from xgboost import XGBClassifier
from IPython.display import Image  
from sklearn.manifold import TSNE
from keras.layers import Input, Dense
from sklearn.decomposition import PCA
from sklearn.tree import export_graphviz
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from keras.models import Sequential, Model
from sklearn.externals.six import StringIO  
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense, Dropout, Input
from sklearn import preprocessing, metrics, model_selection