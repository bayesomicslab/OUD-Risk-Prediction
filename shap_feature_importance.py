import argparse
import json
import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# CLI argument parser
parser = argparse.ArgumentParser(description='Options to parameterize the train and test script.')
parser.add_argument('--dataset', type=str, help='Path to dataset folder')
parser.add_argument('--var_features', type=str, help='File name for selected variants from backward stepwise regression')
parser.add_argument('--rr_geno', type=int, default=10, help='Genotype risk ratio')
parser.add_argument('--rr_mt', type=int, default=1, help='Mobility trace risk ratio')
parser.add_argument('--hyperparams', type=str, help='Path to hyperparameters file')
parser.add_argument('--out', type=str, help='Path to ouput folder')
ARGS, unparsed = parser.parse_known_args()

N = 100

CLUSTER_TYPES = {
    'tsc1': 'outdoors_rec', 'tsc2': 'profess_oth', 'tsc3': 'shop',
    'tsc4': 'food', 'tsc5': 'transport', 'tsc6': 'residence',
    'tsc7': 'university', 'tsc8': 'arts_entert', 'tsc9': 'nightlife',
} 

# Read in feature selection output file
var_features_file = open(ARGS.var_features, 'r')
variant_features = var_features_file.read()
features_list = variant_features.split('\n')
var_features_file.close()

features_list.pop()

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


def write_results(res_list, filename):
    with open('{0}/{1}'.format(ARGS.out, filename), 'w') as outfile:
        json.dump(res_list, outfile, cls=NpEnconder)

def train_and_explain(X_train, y_train, X_test, model, model_name):
    model.fit(X_train, y_train) # fitting the model

    if model_name == 'logit' or model_name == 'svc':
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        return shap_values.values
    else:
        explainer = shap.TreeExplainer(model)
        tree_shap_values = explainer.shap_values(X_test)
        return tree_shap_values[0]
     

def train_models(datasets_path, n_datasets=N):
    
    dataset_filename = 'data_geno_{}_mt_{}'.format(ARGS.rr_geno, ARGS.rr_mt)

    df = pd.read_csv('{0}/{1}_set_0.csv'.format(datasets_path, dataset_filename))
    df.rename(columns=CLUSTER_TYPES)

    df_cols = df.iloc[:, :21].columns.tolist() + features_list
    df_cols = [col.upper() for col in df_cols]

    shap_values_dict = {
        'df_cols': df_cols,
        'logit':[],
        'svc':[],
        'knn':[],
        'dt':[],
        'rf':[],
        'ada':[],
        'gbc':[],
    }

    X_test_dict = {
        'df_cols': df_cols,
        'X_test': []
    }

    for i in range(n_datasets):
        X_train, X_test, y_train, _ = split('{0}/{1}_set_{2}.csv'.format(datasets_path, dataset_filename, i))

        # Merged
        X_train = pd.concat([X_train.iloc[:, :21], X_train[features_list]], axis=1)
        X_test = pd.concat([X_test.iloc[:, :21], X_test[features_list]], axis=1)

        # Normalizing the data
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Only doing interpretation for merged models
        model = read_hyperparams_file('{folder}/models_merged.json'.format(folder=ARGS.hyperparams))

        # LOGISTIC
        lr_model = LogisticRegression(
            C=model['logit']['clf__C'], 
            penalty=model['logit']['clf__penalty'],
            solver=model['logit']['clf__solver'], 
            max_iter=50000
        )
        shap_values_dict['logit'].append(train_and_explain(X_train, y_train, X_test, lr_model, 'logit')) 

        # SVM
        svc_model = LinearSVC(
            C=model['svc']['clf__C'], 
            tol=model['svc']['clf__tol'],
            max_iter=50000
        )
        shap_values_dict['svc'].append(train_and_explain(X_train, y_train, X_test, svc_model, 'svc'))

        # DECISION TREE
        dt_model = DecisionTreeClassifier(
            max_depth=model['dt']['clf__max_depth'], 
            max_features=model['dt']['clf__max_features'],
            min_samples_leaf=model['dt']['clf__min_samples_leaf'], 
            criterion=model['dt']['clf__criterion']
        )
        shap_values_dict['dt'].append(train_and_explain(X_train, y_train, X_test, dt_model, 'dt'))

        # RANDOM FOREST
        rf_model = RandomForestClassifier(
            n_estimators=model['rf']['clf__n_estimators'], 
            max_features=model['rf']['clf__max_features'],
            max_depth=model['rf']['clf__max_depth'], 
            min_samples_split=model['rf']['clf__min_samples_split'], 
            bootstrap=model['rf']['clf__bootstrap'], 
            criterion=model['rf']['clf__criterion']
        )
        shap_values_dict['rf'].append(train_and_explain(X_train, y_train, X_test, rf_model, 'rf'))

        # GBC
        gbc_model = GradientBoostingClassifier(
            learning_rate=model['gbc']['clf__learning_rate'], 
            max_depth=model['gbc']['clf__max_depth'],
            max_features=model['gbc']['clf__max_features'], 
            subsample=model['gbc']['clf__subsample'],
            n_estimators=model['gbc']['clf__n_estimators']
        )
        shap_values_dict['gbc'].append(train_and_explain(X_train, y_train, X_test, gbc_model, 'gbc'))

        X_test_dict['X_test'].append(X_test.tolist())

    return shap_values_dict, X_test_dict

def aggregate_results(values_dict):
    dict_keys = list(values_dict.keys())[1:]

    for key in dict_keys:
        values_dict[key] = np.stack([np.array(a) for a in values_dict[key]], axis=0)
        values_dict[key] = np.mean(values_dict[key], axis=0)
    
    return values_dict

def main():
    shap_values_dictionary, X_test_dict  = train_models(ARGS.dataset)

    shap_values_dictionary = aggregate_results(shap_values_dictionary)
    write_results(shap_values_dictionary, 'agg_shap_values_geno_{}_mt_{}.json'.format(ARGS.rr_geno, ARGS.rr_mt))

    agg_X_test_ = aggregate_results(X_test_dict)
    write_results(agg_X_test_, 'agg_X_test_geno_{}_mt_{}.json'.format(ARGS.rr_geno, ARGS.rr_mt))
    

if __name__ == '__main__':
    main()