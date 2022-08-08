'''
@author: Sybille M. Legitime

Use backward stepwise regression to remove collinear features
'''

import argparse
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='File name for merged data')
parser.add_argument('--out', type=str, help='Path of output file')
ARGS, unparsed = parser.parse_known_args()

'''~~ HELPER FUNCTIONS ~~'''

# Backward feature elimination
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

def read_and_split(filename):
    data = pd.read_csv(filename)
    # get X and y
    X = data.iloc[:, :-1].astype(float)
    y = data.iloc[:, -1]

    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

def select_features(filename):
    # split train and test
    X_train,_ , y_train,_ = read_and_split(filename)

    genotype_train_data = X_train.iloc[:, 21:]

    included = backward_regression(genotype_train_data, y_train, verbose=False)

    return included

def main():
    selected_features = select_features(ARGS.filename)
    textfile = open(ARGS.out, 'w')
    for element in selected_features:
        textfile.write(element + "\n")
    textfile.close()

if __name__ == "__main__":
    main()