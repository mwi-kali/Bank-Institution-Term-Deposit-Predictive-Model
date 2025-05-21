import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
BANK_FULL = os.path.join(DATA_DIR, 'bank-full.csv')
BANK_ADD = os.path.join(DATA_DIR, 'bank-additional-full.csv')
TUNED_PARAMS_PATH = os.path.join(DATA_DIR, 'tuned_parameters.json')
PREP_DIR = os.path.join(DATA_DIR) 
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Random seed
RANDOM_STATE = 42

# Ordered categories
MONTH_ORDER = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
DAY_ORDER = ['mon','tue','wed','thu','fri']
EDU_ORDER = ['university.degree','high.school','basic.9y','professional.course', 'basic.4y','basic.6y','unknown','illiterate']
MARITAL_ORDER = ['married','single','divorced','unknown']
POUTCOME_ORDER = ['success','failure','nonexistent']
TERNARY_ORDER = ['yes','no','unknown']
BINARY_ORDER = ['yes','no']
