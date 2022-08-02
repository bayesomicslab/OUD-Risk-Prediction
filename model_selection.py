import argparse
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import uniform
from scipy.stats import randint
from scipy.stats import truncnorm
from sklearn.model_selection import GridSearchCV, RepeatedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Perform model selection (random search CV) - merged data, mobility trace only, variants only
# Return best params for the model -> JSON file

class NpEnconder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    if isinstance(obj, np.floating):
      return float(obj)
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if isinstance(obj, np.bool_):
      return bool(obj)
    return super(NpEnconder, self).default(obj)

# CLI argument parser
parser = argparse.ArgumentParser(description='Options to parameterize model selection.')
parser.add_argument('--merged_file', type=str, help='File name for merged data')
parser.add_argument('--var_features', type=str, help='File name for selected variants from backward stepwise regression')
parser.add_argument('--out', type=str, help='Path of ouput file')
ARGS, unparsed = parser.parse_known_args()

# Model selection
TEST_SIZE = 0.2
SEED = 42
N_ITERS = 60
SCORING_METRIC = 'average_precision'

# Read in merged data set
merged = pd.read_csv(ARGS.merged_file)

# Read in feature selection output file
var_features_file = open(ARGS.var_features, 'r')
variant_features = var_features_file.read()
features_list = variant_features.split('\n')
var_features_file.close()

features_list.pop()

# Train test split
X = merged.iloc[:, :-1].astype(float)
y = merged.iloc[:, -1].astype(int)

# Filter merged dataset with feature-selected variants
X = pd.concat([X.iloc[:, :21], X[features_list]], axis=1)

''' ### PERFORM MODEL SELECTION (RANDOM SEARCH/GRID SEARCH DV) - MERGED DATA, MOBILITY TRACE ONLY, VARIANTS ONLY ###'''

# Cross-validator
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

'''~~ DEFINING SEARCH SPACES ~~'''

# Logistic Regression
lr_pipeline = Pipeline([
  ('scale', StandardScaler()),
  ('clf', LogisticRegression(max_iter=10000))                    
])

lr_space = {
  'clf__solver': ['newton-cg', 'lbfgs', 'liblinear'],
  'clf__penalty': ['l1', 'l2'],
  'clf__C': [0.01, 0.1, 1]
}

lr_search = GridSearchCV(lr_pipeline, lr_space, scoring=SCORING_METRIC, cv=cv, n_jobs=-1)

# SVM
svc_pipeline = Pipeline([
  ('scale', StandardScaler()),
  ('clf', LinearSVC())
])

svm_space = {
  'clf__C': [0.01, 0.1, 1, 10],
  'clf__tol': [1e-2, 1e-3, 1e-4, 1e-5]
}

svm_search = RandomizedSearchCV(svc_pipeline, svm_space, n_iter=N_ITERS, scoring=SCORING_METRIC, cv=cv, random_state=SEED, n_jobs=-1)

# K-Nearest Neighbors
knn_pipeline = Pipeline([
  ('scale', StandardScaler()),
  ('clf', KNeighborsClassifier())
])

knn_space = {
  'clf__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
  'clf__leaf_size': np.arange(30, 50, 5),
  'clf__weights': ['uniform', 'distance'],
  'clf__metric': ['minkowski', 'euclidean', 'manhattan']
}

knn_search = GridSearchCV(knn_pipeline, knn_space, scoring=SCORING_METRIC, cv=cv, n_jobs=-1)

# Decision Tree
dt_pipeline = Pipeline([
  ('scale', StandardScaler()),
  ('clf', DecisionTreeClassifier())
])

dt_space = {
  'clf__max_depth': [3, 5, 10, None],
  'clf__max_features': randint(1, 9),
  'clf__min_samples_leaf': randint(1, 10),
  'clf__criterion': ['gini', 'entropy']
}

dt_search = RandomizedSearchCV(dt_pipeline, dt_space, n_iter=N_ITERS, scoring=SCORING_METRIC, cv=cv, random_state=SEED, n_jobs=-1)

# Random Forest
rf_pipeline = Pipeline([
  ('scale', StandardScaler()),
  ('clf', RandomForestClassifier())
])

rf_space = {
  'clf__n_estimators': randint(25, 500),
  'clf__max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1), # gaussian, with mean 0.25 and sd 0.1, bounded,
  'clf__min_samples_split': uniform(0.01, 0.199),
  'clf__bootstrap': [True, False],
  'clf__max_depth': [3, 5, 10, None],
  'clf__criterion': ['gini', 'entropy']
} 

rf_search = RandomizedSearchCV(rf_pipeline, rf_space, n_iter=N_ITERS, scoring=SCORING_METRIC, cv=cv, random_state=SEED, n_jobs=-1)

# AdaBoost
ada_pipeline = Pipeline([
  ('scale', StandardScaler()),
  ('clf', AdaBoostClassifier())
])

ada_space = {
  'clf__learning_rate': list(np.linspace(0.01,1,10)),
  'clf__n_estimators': randint(100, 650)
}

ada_search = RandomizedSearchCV(ada_pipeline, ada_space, n_iter=N_ITERS, scoring=SCORING_METRIC, cv=cv, random_state=SEED, n_jobs=-1)

# Gradient Boosting Classifier
gbc_pipeline = Pipeline([
  ('scale', StandardScaler()),
  ('clf', GradientBoostingClassifier())
])

gbc_space = {
  "clf__learning_rate": uniform(),
  "clf__max_depth": randint(4, 10),
  "clf__max_features":['auto', "log2","sqrt"],
  "clf__subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
  'clf__n_estimators': randint(100, 1000),
}

gbc_search = RandomizedSearchCV(gbc_pipeline, gbc_space, scoring=SCORING_METRIC, n_iter=N_ITERS, cv=cv, random_state=SEED, n_jobs=-1)

# XGBoost
xgb_pipeline = Pipeline([
  # ('scale', StandardScaler()),
  ('clf', xgb.XGBClassifier(use_label_encoder=False))
])

xgb_space = {
  'clf__gamma': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4, 200],
  'clf__learning_rate': [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7],
  'clf__max_depth': [5,6,7,8,9,10,11,12,13,14],
  'clf__n_estimators': [50,65,80,100,115,130,150],
  'clf__reg_alpha': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200],
  'clf__reg_lambda': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200]
}

xgb_search = RandomizedSearchCV(xgb_pipeline, xgb_space, scoring=SCORING_METRIC, n_iter=N_ITERS, cv=cv, random_state=SEED, n_jobs=-1)

def do_model_selection(X, y):
  selected_models = dict()

  # Logistic Regression
  lr_search.fit(X, y)
  print(lr_search.best_params_)
  selected_models['logit'] = lr_search.best_params_

  # SVM
  svm_search.fit(X, y)
  print(svm_search.best_params_)
  selected_models['svc'] = svm_search.best_params_

  # K-Nearest Neighbors
  knn_search.fit(X, y)
  print(knn_search.best_params_)
  selected_models['knn'] = knn_search.best_params_

  # Decision Tree
  dt_search.fit(X, y)
  print(dt_search.best_params_)
  selected_models['dt'] = dt_search.best_params_

  # Random Forest
  rf_search.fit(X, y)
  print(rf_search.best_params_)
  selected_models['rf'] = rf_search.best_params_

  # AdaBoost
  ada_search.fit(X, y)
  print(ada_search.best_params_)
  selected_models['ada'] = ada_search.best_params_

  # Gradient Boosting Classifier
  gbc_search.fit(X, y)
  print(gbc_search.best_params_)
  selected_models['gbc'] = gbc_search.best_params_

  # XGBoost
  # xgb_search.fit(X, y)
  # print(xgb_search.best_params_)
  # selected_models['xgb'] = xgb_search.best_params_

  return selected_models

def write_file(models_dict, filename, message):
  with open('{0}/{1}.json'.format(ARGS.out, filename), 'w') as outfile:
      json.dump(models_dict, outfile, cls=NpEnconder)
  print('Model params for {0} dataset written.'.format(message))

def main():
  '''~~ MODEL SELECTION - MERGED DATA ~~'''
  models_merged = do_model_selection(X, y)
  write_file(models_merged, 'models_merged', 'merged')

  '''~~ MODEL SELECTION - MOBILITY TRACE DATA ~~'''
  X_mt = X.iloc[:, :21].astype(float)
  models_mt = do_model_selection(X_mt, y)
  write_file(models_mt, 'models_mt', 'mobility trace')

  '''~~ MODEL SELECTION - VARIANTS DATA ~~'''
  X_var = X.iloc[:, 21:].astype(float)
  models_var = do_model_selection(X_var, y)
  write_file(models_var, 'models_var', 'genotype')    

if __name__ == '__main__':
  main()