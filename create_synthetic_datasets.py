import random
import logging
import argparse
import numpy as np
import pandas as pd
from scipy.stats import bernoulli
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF
from imblearn.over_sampling import SMOTEN

# Parse arguments
parser = argparse.ArgumentParser(description='Options for running data generation code.')
parser.add_argument('--comorbidity', type=float, default=1.0, help='Comorbidity level for mobility trace feature sampling')
parser.add_argument('--rr_geno', type=int, default=10, help='Genotype risk ratio')
parser.add_argument('--rr_mt', type=int, default=1, help='Mobility trace risk ratio')
parser.add_argument('--n_sets', type=int, default=100, help='Number of sets per comorbidity/risk ratio configuration')
parser.add_argument('--out', type=str, help='Path of ouput folder')
ARGS, unparsed = parser.parse_known_args()

mobility_trace = pd.read_csv('data/mobility_trace/mobility_trace.csv')
variants_data = pd.read_csv('data/target_final_ext_oud_od.raw', delim_whitespace=True)

'''~~ HELPER METHODS ~~'''
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

# Synthetic minority over-sampling (variants data)
over = SMOTEN(sampling_strategy=1, random_state=0)

'''~~ VARIANTS DATA PROCESSING ~~'''
# Replace all missing values with the covariate's mode
for column in variants_data.columns:
    variants_data[column].fillna(variants_data[column].mode()[0], inplace=True) # imputation with the covariate's mode

mapping = {
    1: 0,
    2: 1
}
map_func = np.vectorize(lambda x: mapping[x] if x in mapping else x)
variants_data['PHENOTYPE'] = map_func(variants_data['PHENOTYPE'])
var_features = variants_data.iloc[:, 6:]
y = np.array(variants_data['PHENOTYPE'])

#  Oversample controls to make them equal the number of cases
var_features, y = over.fit_resample(var_features, y)
variants_data_comb = pd.concat([var_features, pd.DataFrame(y, columns=['case'])], axis=1)
# Separating opioid cases and controls for sampling
var_cases = variants_data_comb[variants_data_comb['case'] == 1]
var_controls = variants_data_comb[variants_data_comb['case'] == 0]

'''~~ MOBILITY TRACE DATA PROCESSING ~~'''
mt_features = mobility_trace.iloc[:, 1:-1]
mt_cases = mobility_trace[mobility_trace['depress'] == 1]
mt_controls= mobility_trace[mobility_trace['depress'] == 0] # separate depression cases and controls

def get_case_geno(index):
    return var_cases.iloc[index, :]

def get_control_geno(index):
    return var_controls.iloc[index, :]

def main():
    # Constants
    COMORB_PROB = ARGS.comorbidity
    RR_GENO = ARGS.rr_geno #(15, 10, 5, inf)
    RR_MT = ARGS.rr_mt #(5, 1, inf)
    N = 1000 # number of case and control samples to feature in the dataset
    N_DATASET = ARGS.n_sets # number of datasets to generate
    LOG_FOLDER_PATH = '{}/logs'.format(ARGS.out)

    '''~~ LOGGING ~~'''
    logging.basicConfig(filename='{0}/merged_geno_{1}_mt_{2}.log'.format(LOG_FOLDER_PATH, RR_GENO, RR_MT), level=logging.DEBUG)

    logging.info('Comorbidity = {}'.format(COMORB_PROB))
    logging.info('Genotype RR = {} | Mobility Trace RR = {}'.format(RR_GENO, RR_MT))


    '''~~ MERGE VARIANTS AND SYNTHETIC MOBILITY TRACE DATA ~~'''
    for i in range(N_DATASET):
        n_cases = 0
        n_controls = 0

        merged = pd.DataFrame([], columns= mobility_trace.columns[1:-1].to_list() + variants_data_comb.columns[:-1].to_list())

        # to avoid sampling with replacement
        index_dict = {
            'geno_case_index': 1,
            'geno_control_index': 1
        }

        while n_cases < N or n_controls < N:
            # Select a case or control genotype at random
            rand_int  = np.random.randint(0,2,1)[0]

            if rand_int == 1:
                random_init_sample = get_case_geno(index_dict['geno_case_index']) 
                index_dict['geno_case_index'] += 1
            else:
                random_init_sample = get_control_geno(index_dict['geno_control_index'])
                index_dict['geno_control_index'] += 1

            # Use comorbidity to determine match (case-case, control-control) or mismatch (case-control, control-case) of the feature sets
            C = bernoulli(COMORB_PROB).rvs(size=1)

            if C[0] == 1: # match condition
                if random_init_sample['case'] == 1:
                    mob_trace = create_synthetic_mt_sample_ecdf(mt_cases)
                    mob_trace.append(1.0)
                else:
                    mob_trace = create_synthetic_mt_sample_ecdf(mt_controls)
                    mob_trace.append(0.0)
            else: # mismatch
                if random_init_sample['case'] == 1:
                    mob_trace = create_synthetic_mt_sample_ecdf(mt_controls)
                    mob_trace.append(0.0)
                else:
                    mob_trace = create_synthetic_mt_sample_ecdf(mt_cases)
                    mob_trace.append(1.0)
            # Concat feature sets
            merged_feature_set = pd.concat([pd.Series(mob_trace, index=mt_controls.columns[1:]), random_init_sample])

            logging.debug('Sampled genotype original case status = {} | Sampled mobility trace original case status = {}'.format(random_init_sample['case'], mob_trace[-1]))

            # Assign case or control status
            rr_combined = 0
            prob = 0
            case_status = 0

            def compute_prob(risk):
                return risk / (1 + risk)

            if merged_feature_set['case'] == 1:
                if merged_feature_set['depress'] == 1: # case-case
                    rr_combined = RR_GENO + RR_MT
                else: # case-control
                    rr_combined = RR_GENO + (1 / RR_MT)
            else:
                if merged_feature_set['depress'] == 1: # control-case
                    rr_combined = (1 / RR_GENO) + RR_MT
                else: # control-control
                    rr_combined = (1 / RR_GENO) + (1 / RR_MT)

            prob = compute_prob(rr_combined)
            case_status = bernoulli(prob).rvs(size=1)[0]

            # Finalize new sample
            merged_feature_set = merged_feature_set.drop(labels=['case', 'depress'])
            merged_feature_set['case'] = case_status
            logging.debug('OVERALL assigned case status = {}'.format(case_status))

            # Update merged dataframe
            merged = pd.concat([merged, merged_feature_set.to_frame().T])

            if case_status == 1:
                n_cases += 1
            else:
                n_controls +=1

        logging.info('# cases = {} | # controls = {}'.format(n_cases, n_controls))

        merged_cases = merged[merged['case'] == 1]
        merged_controls = merged[merged['case'] == 0]

        final_merge = pd.concat([merged_cases.sample(n=N, random_state=42), merged_controls.sample(n=N, random_state=42) ], axis=0) # select same number of case samples and control samples for final dataset

        final_merge.to_csv('{out}/geno_{rr_geno}_mt_{rr_mt}/data_geno_{rr_geno}_mt_{rr_mt}_set_{index}.csv'.format(
            out=ARGS.out, rr_geno=RR_GENO, rr_mt=RR_MT, index=i), index=False)

        logging.info('Dataset # {} created.'.format(i))
        logging.info('---------------------------------------------------------------------------')

    print('DATASETS FOR COMORBIDITY = {} | RR_GENO = {} | RR_MT = {} ARE CREATED.'.format(COMORB_PROB, RR_GENO, RR_MT))

if __name__ == '__main__':
    main()
