import random
import argparse
import numpy as np
import pandas as pd
from scipy.stats import bernoulli
import statsmodels.api as sm
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF
from imblearn.over_sampling import SMOTEN

# TODO: Add needed arguments to be passed to command line
# Parse arguments
parser = argparse.ArgumentParser(description='Options for running data generation code.')
parser.add_argument('--comorbidity', type=float, default=1.0, help='Comorbidity level for mobility trace feature sampling')
parser.add_argument('--n_sets', type=int, default=1, help='Number of sets per comorbidity level')
parser.add_argument('--out', type=str, help='Path in which to ouput file')
ARGS, unparsed = parser.parse_known_args()

'''~~HELPER METHODS~~'''
# Generate inverted ECDF with linear interpolation
def get_inverted_cdf(var):
    var_edf = ECDF(var)
    slope_changes = sorted(set(var))

    var_edf_values_at_slope_changes = [var_edf(item) for item in slope_changes]
    inverted_edf = interp1d(var_edf_values_at_slope_changes, slope_changes, fill_value='extrapolate')

    return inverted_edf

# Will generate a synthetic mobility trace record by random sampling of mobility trace features FROM EMPIRICAL DISTRIBUTION
def create_synthetic_mt_sample_ecdf(df):
    filtered_df = df.iloc[:, 1:-1]
    inverteds = [get_inverted_cdf(filtered_df[col].values) for col in filtered_df.columns]
    sample_record = [inverteds[i](random.uniform(0, 1)) for i in range(len(filtered_df.columns))]
    return sample_record

# Feature selection of variants through Backward elimination
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

'''~~DATA PREPROCESSING - MOBILITY TRACE DATA~~'''
mobility_trace = pd.read_csv('data/mobility_trace.csv')

depressed = mobility_trace[mobility_trace['depress'] == 1]
non_depressed = mobility_trace[mobility_trace['depress'] == 0]

'''~~DATA PREPROCESSING - VARIANTS DATA~~'''
# Variants data
variants_data = pd.read_csv('data/target_final_ext_oud_od.raw', delim_whitespace=True)

# Replace all missing values with the covariate's mode
for column in variants_data.columns:
    variants_data[column].fillna(variants_data[column].mode()[0], inplace=True)

variants_data = variants_data.astype(int)

X = variants_data.iloc[:, 6:39]
y = np.array(variants_data['PHENOTYPE'])

# Change 'case' 'control' labels 
mapping = {
    1: 0,
    2: 1
}

mapping_func = np.vectorize(lambda x: mapping[x] if x in mapping else x)
y = mapping_func(y)

# Over-sampling TODO: specify sampling strategy with SMOTE
over = SMOTEN(sampling_strategy=1, random_state=0)
X, y = over.fit_resample(X, y)

# Recombine features and labels
variants_oversampled = pd.concat([X, pd.DataFrame(y, columns=['case'])], axis=1)

# Create valid list of variants data features
variants = variants_oversampled.columns
variants = variants.to_list()
variants.pop()

'''~~MERGE VARIANTS AND MOBILITY TRACE DATA~~'''
p = ARGS.comorbidity
ber = bernoulli(p)

for set in range(ARGS.n_sets):
    variants_with_synthetic = pd.DataFrame()

    filtered_columns = mobility_trace.columns.to_list()
    filtered_columns.pop(0)
    filtered_columns.pop(-1)

    for index, row in variants_oversampled.iterrows():
        bernoulli_res = ber.rvs(size=1)
        if row['case'] == 1:
            mob_features = create_synthetic_mt_sample_ecdf(depressed) if bernoulli_res[0] == 1 else create_synthetic_mt_sample_ecdf(non_depressed)
        else:
            mob_features = create_synthetic_mt_sample_ecdf(non_depressed) if bernoulli_res[0] == 1 else create_synthetic_mt_sample_ecdf(depressed)
        mob_feature_set = pd.DataFrame([mob_features], index = [index], columns=filtered_columns)
        mob_feature_set[variants_oversampled.columns] = pd.DataFrame([np.concatenate([ [row[variant] for variant in variants], [row['case']] ])], 
                                                                    index=mob_feature_set.index)
        variants_with_synthetic = pd.concat([variants_with_synthetic, mob_feature_set])
    variants_with_synthetic.to_csv(ARGS.out, index=False)
