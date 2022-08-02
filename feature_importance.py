import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def print_feature_importance(features, model_name, importance):
    print('Feature Importance - {}'.format(model_name))
    for feat, importance in zip(features, importance):
        print('Feature: {}, Score: {:.5f}'.format(feat, importance))

def make_dist_grid(features, traces, filename, n_rows=7, n_cols=3, title='Distribution of Mobility Trace Features (X_train)'):
    # Create grid
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=features)

    # Append to grid
    i = 0
    for row in range(1, 8):
        for col in range(1, 4):
            fig.append_trace(traces[i], row, col)
            i += 1

    fig.update_layout(autosize=False, width= 1500, height=850, title=title)
    fig.write_image(filename)

def generate_feature_importance(df):
    X = df.iloc[:, :21]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)

    mt_features = X_train.columns[:21]
    traces = [go.Histogram(x=X_train[feature].values) for feature in mt_features]
    make_dist_grid(mt_features, traces, 'feat_dist_original.png')

    # Preprocessing
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(sc.transform(X_test), columns=X.columns)
    
    traces_z = [go.Histogram(x=X_train[feature].values) for feature in mt_features]
    make_dist_grid(mt_features, traces_z, 'feat_dist_z.png', title='Distribution of Mobility Trace Features (X_train, Normalized)')

    logit = LogisticRegression(
        C=3.3981724150105963,
        penalty='l2',
        solver='liblinear',
        max_iter=200
    )
    logit.fit(X_train, y_train)

    logit_importance = logit.coef_[0]

    # Decision Tree
    dt_merged_dict = dict(
        criterion='gini',
        max_depth=None,
        max_features=8,
        min_samples_leaf=8
    )
    dt_mt_dict = dict(
        criterion='entropy',
        max_depth=10,
        max_features=4,
        min_samples_leaf=9
    )

    dt = DecisionTreeClassifier(
        criterion=dt_mt_dict['criterion'],
        max_depth=dt_mt_dict['max_depth'],
        max_features=dt_mt_dict['max_features'],
        min_samples_leaf=dt_mt_dict['min_samples_leaf']
    )
    dt.fit(X_train, y_train)

    dt_importance = dt.feature_importances_

    # Permutation Feature Importance
    svc = SVC(
        C=4.570563099801452,
        gamma=0.006251373574521747,
        kernel='rbf',
        kernel='linear'
    )
    linear_svc = SVC(kernel='linear', probability=True, max_iter=200)
    linear_svc.fit(X_train, y_train)

    res = permutation_importance(linear_svc, X_train, y_train, scoring='accuracy')
    perm_importance = res.importances_mean

    return logit_importance, dt_importance, perm_importance


def main():
    agg_dict = {
        'logit': [],
        'dt': [],
        'perm': []
    }

    for i in range(10):
        data = pd.read_csv('./sets/dataset_1.0_{}.csv'.format(i))

        logit_imp, dt_imp, perm_imp = generate_feature_importance(data)
        agg_dict['logit'].append(logit_imp)
        agg_dict['dt'].append(dt_imp)
        agg_dict['perm'].append(perm_imp)

    agg_dict['logit'] = np.mean(np.asarray(agg_dict['logit']), axis=0)
    agg_dict['dt'] = np.mean(np.asarray(agg_dict['dt']), axis=0)
    agg_dict['perm'] = np.mean(np.asarray(agg_dict['perm']), axis=0)
    
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,6))
    df = pd.read_csv('./sets/dataset_1.0_0.csv')
    mt_cols = df.iloc[:, :21].columns
    merged_cols = df.iloc[:, :-1].columns

    mt_ticks = np.arange(len(mt_cols))
    merged_ticks = np.arange(len(merged_cols))

    df = pd.read_csv('./sets/dataset_1.0_0.csv')

    ax1.bar([x for x in range(len(agg_dict['logit']))], agg_dict['logit'])
    ax1.set_title('Logit Coefficients')
    ax1.set_xticks(mt_ticks)
    ax1.set_xticklabels(mt_cols, rotation=50)

    ax2.bar([x for x in range(len(agg_dict['dt']))], agg_dict['dt'])
    ax2.set_title('Decision Tree Feature Importance')
    ax2.set_xticks(mt_ticks)
    ax2.set_xticklabels(mt_cols, rotation=50)

    ax3.bar([x for x in range(len(agg_dict['perm']))], agg_dict['perm'])
    ax3.set_title('Permutation Feature Importance')
    ax3.set_xticks(mt_ticks)
    ax3.set_xticklabels(mt_cols, rotation=50)

    fig.savefig('feature_importance_mt_tuned_1.png')

if __name__ == '__main__':
    main()
