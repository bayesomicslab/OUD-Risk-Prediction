from time import time
import pandas as pd
import numpy as np
from scipy.stats import loguniform, uniform
from scipy.stats import truncnorm
from scipy.stats import randint
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
from sklearn.ensemble import GradientBoostingClassifier

TEST_SIZE = 0.2
SEED = 42
N_ITERS = 60
SCORING_METRIC = 'average_precision'

# Script is already oversampled AND feature selected
merged = pd.read_csv('merged_variants_mt_data.csv')

# Generate train-test sets
def generate_train_test(data: 'pd.DataFrame', test_size: float, seed: int, dataset_type='combined'):
  X = pd.DataFrame()
  y = data.iloc[:, -1]

  if dataset_type == 'mobility_trace':
    X = data.iloc[:, :21].astype(float)
  elif dataset_type == 'variants':
    X = data.iloc[:, 21:].astype(float)
  else:
    X = data.iloc[:, :-1].astype(float)

  return train_test_split(X, y, test_size=test_size, random_state=seed)

# Combined data set
X_train, X_test, y_train, y_test = generate_train_test(merged, TEST_SIZE, SEED)

# Mobility trace data
X_train_mt, X_test_mt, y_train_mt, y_test_mt = generate_train_test(merged, TEST_SIZE, SEED, 'mobility_trace')

# Variants data
X_train_var, X_test_var, y_train_var, y_test_var = train_test_split(merged, TEST_SIZE, SEED, 'variants')

def display_search_stats(clf, t0):
  print('done in {0:.3f}s'.format(time() - t0))
  print('Best estimator found by randomized search:')
  print(clf.best_estimator_)

def classify_and_evaluate(X_test, y_test, clf):
  y_pred = clf.predict(X_test)
  print(classification_report(y_test, y_pred))

  return classification_report(y_test, y_pred)

def write_file(class_res, filename):
  f = open(filename, 'w')
  f.write(class_res)
  f.close()

# TODO : return scores to be written in report
def display_auc_scores(X_test, y_test, clf):
    # predictions
    y_hat = clf.predict_proba(X_test)
    # get probabilities only for positive class
    pos_probs = y_hat[:, 1]

    # calculate model pr curve
    precision, recall, _ = precision_recall_curve(y_test, pos_probs)

    print('ROC AUC Score:{}'.format(roc_auc_score(y_test, pos_probs)))
    print('Precision-Recall AUC:{}'.format(auc(recall, precision)))


# Define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)

'''
~~~~LOGISTIC REGRESSION~~~
'''

lr_pipeline = Pipeline([
  ('scale', StandardScaler()),
  ('clf', LogisticRegression(class_weight='balanced'))                    
])

# define search space
lr_space = {
    'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
    'clf__penalty': ['l1', 'l2', 'elasticnet'],
    'clf__C': loguniform(1e-5, 10)
}

# define search
lr_search = RandomizedSearchCV(lr_pipeline, lr_space, n_iter=N_ITERS, 
                               scoring=SCORING_METRIC, cv=cv, random_state=SEED, n_jobs=-1)

# COMBINED DATA SET: VARIANTS AND MOBILITY TRACE FEATURES
lr_search.fit(X_train, y_train)
lr_res_combined = classify_and_evaluate(X_test, y_test, lr_search)
write_file(lr_res_combined, 'log_reg_combined.txt')

# MOBILITY TRACE FEATURES ONLY
lr_search.fit(X_train_mt, y_train_mt)
lr_res_mt = classify_and_evaluate(X_test_mt, y_test_mt, lr_search)
write_file(lr_res_mt, 'log_reg_mt.txt')

# VARIANTS DATA ONLY
lr_search.fit(X_train_var, y_train_var)
lr_res_var = classify_and_evaluate(X_test_var, y_test_var, lr_search)
write_file(lr_res_var, 'log_reg_var.txt')


'''
~~~~SVM~~~
'''
svc_pipeline = Pipeline([
  ('scale', StandardScaler()),
  ('clf', SVC(class_weight='balanced', probability=True))
])

svm_space = {
    'clf__C': loguniform(1e-3, 1e2),
    'clf__gamma': loguniform(1e-4, 1e-1),
    'clf__kernel': ['rbf']
}

svm_search = RandomizedSearchCV(svc_pipeline, svm_space, n_iter=N_ITERS, 
                                scoring=SCORING_METRIC, cv=cv, random_state=SEED, n_jobs=-1)

# COMBINED DATA SET: VARIANTS AND MOBILITY TRACE FEATURES
svm_search.fit(X_train, y_train)
svm_res_combined = classify_and_evaluate(X_test, y_test, svm_search)
write_file(svm_res_combined, 'svc_combined.txt')

# MOBILITY TRACE FEATURES ONLY
svm_search.fit(X_train_mt, y_train_mt)
svm_res_mt = classify_and_evaluate(X_test_mt, y_test_mt, svm_search)
write_file(svm_res_mt, 'svc_mt.txt')

# VARIANTS DATA ONLY
svm_search.fit(X_train_var, y_train_var)
svm_res_var = classify_and_evaluate(X_test_var, y_test_var, svm_search)
write_file(svm_res_var, 'svc_var.txt')


'''
~~~~K-NEAREST NEIGHBORS~~~
'''
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

# COMBINED DATA SET: VARIANTS AND MOBILITY TRACE FEATURES
knn_search.fit(X_train, y_train)
knn_res_combined = classify_and_evaluate(X_test, y_test, knn_search)
write_file(knn_res_combined, 'knn_combined.txt')

# MOBILITY TRACE FEATURES ONLY
knn_search.fit(X_train_mt, y_train_mt)
knn_res_mt = classify_and_evaluate(X_test_mt, y_test_mt, knn_search)
write_file(knn_res_mt, 'knn_mt.txt')

# VARIANTS DATA ONLY
knn_search.fit(X_train_var, y_train_var)
knn_res_var = classify_and_evaluate(X_test_var, y_test_var, knn_search)
write_file(knn_res_var, 'knn_var.txt')


'''
~~~~DECISION TREE~~~
'''
dt_pipeline = Pipeline([
  ('scale', StandardScaler()),
  ('clf', DecisionTreeClassifier(class_weight='balanced'))
])

dt_space = {
    'clf__max_depth': [3, 5, 10, None],
    'clf__max_features': randint(1, 9),
    'clf__min_samples_leaf': randint(1, 10),
    'clf__criterion': ['gini', 'entropy']
}

dt_search = RandomizedSearchCV(dt_pipeline, dt_space, n_iter=N_ITERS,
                               scoring=SCORING_METRIC, cv=cv, random_state=SEED, n_jobs=-1)

# COMBINED DATA SET: VARIANTS AND MOBILITY TRACE FEATURES
dt_search.fit(X_train, y_train)
dt_res_combined = classify_and_evaluate(X_test, y_test, dt_search)
write_file(dt_res_combined, 'decision_tree_combined.txt')

# MOBILITY TRACE FEATURES ONLY
dt_search.fit(X_train_mt, y_train_mt)
dt_res_mt = classify_and_evaluate(X_test_mt, y_test_mt, dt_search)
write_file(dt_res_mt, 'decision_tree_mt.txt')

# VARIANTS DATA ONLY
dt_search.fit(X_train_var, y_train_var)
dt_res_var = classify_and_evaluate(X_test_var, y_test_var, dt_search)
write_file(dt_res_var, 'decision_tree_var.txt')



'''
~~~~RANDOM FOREST~~~
'''
rf_pipeline = Pipeline([
  ('scale', StandardScaler()),
  ('clf', RandomForestClassifier(class_weight='balanced_subsample'))
])

rf_space = {
    'clf__n_estimators': randint(25, 500),
    'clf__max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1), # gaussian, with mean 0.25 and sd 0.1, bounded,
    'clf__min_samples_split': uniform(0.01, 0.199),
    'clf__bootstrap': [True, False],
    'clf__max_depth': [3, 5, 10, None],
    'clf__criterion': ['gini', 'entropy']
} 

rf_search = RandomizedSearchCV(rf_pipeline, rf_space, n_iter=N_ITERS, scoring=SCORING_METRIC, 
                               cv=cv, random_state=SEED, n_jobs=-1)

# COMBINED DATA SET: VARIANTS AND MOBILITY TRACE FEATURES
rf_search.fit(X_train, y_train)
rf_res_combined = classify_and_evaluate(X_test, y_test, rf_search)
write_file(rf_res_combined, 'random_forest_combined.txt')

# MOBILITY TRACE FEATURES ONLY
rf_search.fit(X_train_mt, y_train_mt)
rf_res_mt = classify_and_evaluate(X_test_mt, y_test_mt, rf_search)
write_file(rf_res_mt, 'random_forest_mt.txt')

# VARIANTS DATA ONLY
rf_search.fit(X_train_var, y_train_var)
rf_res_var = classify_and_evaluate(X_test_var, y_test_var, rf_search)
write_file(rf_res_var, 'random_forest_var.txt')


'''
~~~~ADA BOOST~~~
'''
ada_pipeline = Pipeline([
  ('scale', StandardScaler()),
  ('clf', AdaBoostClassifier())
])

ada_space = {
    'clf__learning_rate': list(np.linspace(0.01,1,10)),
    'clf__n_estimators': randint(100, 650)
}

ada_search = RandomizedSearchCV(ada_pipeline, ada_space, n_iter=N_ITERS, scoring=SCORING_METRIC, 
                               cv=cv, random_state=SEED, n_jobs=-1)

# COMBINED DATA SET: VARIANTS AND MOBILITY TRACE FEATURES
ada_search.fit(X_train, y_train)
ada_res_combined = classify_and_evaluate(X_test, y_test, ada_search)
write_file(ada_res_combined, 'ada_combined.txt')

# MOBILITY TRACE FEATURES ONLY
ada_search.fit(X_train_mt, y_train_mt)
ada_res_mt = classify_and_evaluate(X_test_mt, y_test_mt, ada_search)
write_file(ada_res_mt, 'ada_mt.txt')

# VARIANTS DATA ONLY
ada_search.fit(X_train_var, y_train_var)
ada_res_var = classify_and_evaluate(X_test_var, y_test_var, ada_search)
write_file(ada_res_var, 'ada_var.txt')


'''
~~~~GRADIENT BOOSTING CLASSIFIER~~~
'''

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

gbc_search = RandomizedSearchCV(gbc_pipeline, gbc_space, scoring='average_precision', n_iter=60, 
                               cv=cv, random_state=42, n_jobs=-1)

# COMBINED DATA SET: VARIANTS AND MOBILITY TRACE FEATURES
gbc_search.fit(X_train, y_train)
gbc_res_combined = classify_and_evaluate(X_test, y_test, gbc_search)
write_file(gbc_res_combined, 'gbc_combined.txt')

# MOBILITY TRACE FEATURES ONLY
gbc_search.fit(X_train_mt, y_train_mt)
gbc_res_mt = classify_and_evaluate(X_test_mt, y_test_mt, gbc_search)
write_file(gbc_res_mt, 'gbc_mt.txt')

# VARIANTS DATA ONLY
gbc_search.fit(X_train_var, y_train_var)
gbc_res_var = classify_and_evaluate(X_test_var, y_test_var, gbc_search)
write_file(gbc_res_var, 'gbc_var.txt')