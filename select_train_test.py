import argparse
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# CLI argument parser
parser = argparse.ArgumentParser(description='Options to parameterize the train and test script.')
parser.add_argument('--dataset', type=str, help='Path to dataset folder')
parser.add_argument('--rr_geno', type=int, default=10, help='Genotype risk ratio')
parser.add_argument('--rr_mt', type=int, default=1, help='Mobility trace risk ratio')
parser.add_argument('--hyperparams', type=str, help='Path to hyperparameters file')
parser.add_argument('--out', type=str, help='Path to ouput folder')
ARGS, unparsed = parser.parse_known_args()

N = 100

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

'''Read hyperparameter files'''
def read_hyperparams_file(hyperparams_file):
    with open(hyperparams_file, 'r') as json_file:
        model_hyperparams = json.load(json_file)
    return model_hyperparams

'''Read datasets files'''
def read_dataset(path):
    datasets = dict()
    for i in range(N):
        datasets[i] = pd.read_csv('{0}/'.format(path))
    return datasets

'''Split dataset into train and test'''
def split(dataset):
    dataset = pd.read_csv(dataset)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    return train_test_split(X, y, test_size=0.2, random_state=42)


'''Feature selection'''
def backward_regression(X, y, threshold_out=0.05, verbose=True):
    included = list(X.columns)
    while True:
        changed = False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefficients except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {0} with p-value {1}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

def evaluate(X_test, y_test, clf):
    # Predict
    if hasattr(clf, 'predict_proba'):
        y_hat = clf.predict_proba(X_test) # predicting probability for each class
        pos_class = y_hat[:, 1]
    else:
        y_hat = clf.decision_function(X_test) # predicting confidence scores
        pos_class = y_hat
    y_pred = clf.predict(X_test)

    # Get ROC curve dict and AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test.ravel(), pos_class)
    roc_auc = roc_auc_score(y_test, pos_class)

    # Get PR curve dict and AUC
    precision = dict()
    recall = dict()
    pr_auc = dict()
    precision, recall, _ = precision_recall_curve(y_test.ravel(), pos_class)
    pr_auc = average_precision_score(y_test, pos_class)

    # Get accuracy score
    acc = accuracy_score(y_test, y_pred) 

    return (fpr, tpr, roc_auc), (precision, recall, pr_auc), acc

# Output -> (tpr, fpr, roc_auc), (precision, recall, pr_auc), accuracy
def write_results(res_list, filename):
    with open('{0}/{1}'.format(ARGS.out, filename), 'w') as outfile:
        json.dump(res_list, outfile, cls=NpEnconder)


'''~~ TRAINING ~~'''

def evaluate_model(X_test, y_test, model):
    (fpr, tpr, roc_auc), (precision, recall, pr_auc), acc = evaluate(X_test, y_test, model)

    return dict(fpr=fpr, tpr=tpr, roc_auc=roc_auc, precision=precision, recall=recall, pr_auc=pr_auc, accuracy=acc)


def train_and_evaluate(X_train, y_train, X_test, y_test, model, model_name, dataset_type, set_num):
    model.fit(X_train, y_train) # fitting the model

    eval_res_train = evaluate_model(X_train, y_train, model) # training set results
    eval_res_test = evaluate_model(X_test, y_test, model) # test set results

    write_results(eval_res_train, 'train_res/eval_output_{0}_{1}_train_{2}.json'.format(dataset_type, model_name, set_num))
    write_results(eval_res_test, 'test_res/eval_output_{0}_{1}_test_{2}.json'.format(dataset_type, model_name, set_num))


def train_models(datasets_path, n_datasets=N):
    dataset_filename = 'data_geno_{}_mt_{}'.format(ARGS.rr_geno, ARGS.rr_mt)

    for i in range(n_datasets):
        X_train, X_test, y_train, y_test = split('{0}/{1}_set_{2}.csv'.format(datasets_path, dataset_filename, i))

        '''~~ FEATURE SELECTION ~~'''
        variants = X_train.iloc[:, 21:].astype(int)
        included = backward_regression(variants, y_train, verbose=False)

        '''Write file with selected feature variants'''
        textfile = open('{0}/selected_features_{1}.txt'.format(ARGS.out, i), 'w')
        for element in included:
            textfile.write(element + "\n")
        textfile.close()

        # Merged
        X_train = pd.concat([X_train.iloc[:, :21], X_train[included]], axis=1)
        X_test = pd.concat([X_test.iloc[:, :21], X_test[included]], axis=1)

        # Mobility trace
        X_train_mt = X_train.iloc[:, :21].astype(float)
        X_test_mt = X_test.iloc[:, :21].astype(float)

        # Variants
        X_train_var = X_train.iloc[:, 21:]
        X_test_var = X_test.iloc[:, 21:]

        # Normalizing the data
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        sc_mt = StandardScaler()
        X_train_mt = sc_mt.fit_transform(X_train_mt)
        X_test_mt = sc_mt.transform(X_test_mt)

        sc_var = StandardScaler()
        X_train_var = sc_var.fit_transform(X_train_var)
        X_test_var = sc_var.transform(X_test_var)

        datasets = {
            'merged': [X_train, X_test],
            'mt': [X_train_mt, X_test_mt],
            'var': [X_train_var, X_test_var],
        }

        for key, data_array in datasets.items():

            filename = 'models_{0}.json'.format(key)

            model = read_hyperparams_file('{folder}/{file}'.format(folder=ARGS.hyperparams, file=filename))

            # LOGISTIC
            lr_model = LogisticRegression(C=model['logit']['clf__C'], penalty=model['logit']['clf__penalty'],
                    solver=model['logit']['clf__solver'], max_iter=10000)
            train_and_evaluate(data_array[0], y_train, data_array[1], y_test, lr_model, 'logit', key, i)

            # SVM
            svc_model = LinearSVC(C=model['svc']['clf__C'], tol=model['svc']['clf__tol'])
            train_and_evaluate(data_array[0], y_train, data_array[1], y_test, svc_model, 'svc', key, i)

            # KNN
            knn_model = KNeighborsClassifier(n_neighbors=model['knn']['clf__n_neighbors'], leaf_size=model['knn']['clf__leaf_size'],
                        weights=model['knn']['clf__weights'], metric=model['knn']['clf__metric'])
            train_and_evaluate(data_array[0], y_train, data_array[1], y_test, knn_model, 'knn', key, i)

            # DECISION TREE
            dt_model = DecisionTreeClassifier(max_depth=model['dt']['clf__max_depth'], max_features=model['dt']['clf__max_features'],
                        min_samples_leaf=model['dt']['clf__min_samples_leaf'], criterion=model['dt']['clf__criterion'])
            train_and_evaluate(data_array[0], y_train, data_array[1], y_test, dt_model, 'dt', key, i)

            # RANDOM FOREST
            rf_model = RandomForestClassifier(n_estimators=model['rf']['clf__n_estimators'], max_features=model['rf']['clf__max_features'],
                        max_depth=model['rf']['clf__max_depth'], min_samples_split=model['rf']['clf__min_samples_split'], 
                        bootstrap=model['rf']['clf__bootstrap'], criterion=model['rf']['clf__criterion'])
            train_and_evaluate(data_array[0], y_train, data_array[1], y_test, rf_model, 'rf', key, i)
            
            # ADABOOST
            ada_model = AdaBoostClassifier(learning_rate=model['ada']['clf__learning_rate'], n_estimators=model['ada']['clf__n_estimators'])
            train_and_evaluate(data_array[0], y_train, data_array[1], y_test, ada_model, 'ada', key, i)

            # GBC
            gbc_model = GradientBoostingClassifier(learning_rate=model['gbc']['clf__learning_rate'], max_depth=model['gbc']['clf__max_depth'],
                        max_features=model['gbc']['clf__max_features'], subsample=model['gbc']['clf__subsample'],
                        n_estimators=model['gbc']['clf__n_estimators'])
            train_and_evaluate(data_array[0], y_train, data_array[1], y_test, gbc_model, 'gbc', key, i)

def main():
    train_models(ARGS.dataset)

if __name__ == '__main__':
    main()


