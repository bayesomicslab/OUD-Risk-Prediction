import argparse
import glob
import json
import numpy as np

# CLI argument parser
parser = argparse.ArgumentParser(description='Options to parameterize the train and test script.')
parser.add_argument('--dataset', type=str, help='Path to dataset folder')
parser.add_argument('--comorb', type=str, help='Comorbidity code')
parser.add_argument('--out', type=str, help='Path to ouput folder')
ARGS, unparsed = parser.parse_known_args()

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

def agg_curves(data_dict):
  metrics_array_roc = ['fpr', 'tpr']
  for metric in metrics_array_roc:
    length = max(map(len, data_dict[metric]))
    
    data_dict[metric] = np.concatenate([np.array([np.pad(a, (0, length - len(a)), 'maximum')]) for a in data_dict[metric]], axis=0)
    data_dict[metric] = np.mean(data_dict[metric], axis=0)

  metrics_array_pr = ['precision', 'recall']
  for metric in metrics_array_pr:
    length = max(map(len, data_dict[metric]))

    data_dict[metric] = np.concatenate([np.array([np.pad(a, (0, length - len(a)), 'linear_ramp')]) for a in data_dict[metric]], axis=0)
    data_dict[metric] = np.mean(data_dict[metric], axis=0)

  data_dict['roc_auc'] = [np.mean(np.array(data_dict['roc_auc'])), np.std(np.array(data_dict['roc_auc']))]
  data_dict['pr_auc'] = [np.mean(np.array(data_dict['pr_auc'])), np.std(np.array(data_dict['roc_auc']))]
  data_dict['accuracy'] = [np.mean(np.array(data_dict['accuracy'])), np.std(np.array(data_dict['accuracy']))]

  return data_dict

def create_agg_curves(d_type, model):
  data_dict = {
      'fpr': [],
      'tpr': [],
      'precision': [],  
      'recall': [],
      'roc_auc': [],
      'pr_auc': [],
      'accuracy': []
    }

  for name in glob.glob('{folder_prefix}/test_res/eval_output_{dtype}_{model}_test_*'.format(
    folder_prefix=ARGS.dataset,
    dtype=d_type,
    model=model)):
    print(name)
    with open(name, 'r') as json_file:
      eval_res = json.load(json_file)
      data_dict['fpr'].append(eval_res['fpr'])
      data_dict['tpr'].append(eval_res['tpr'])
      data_dict['precision'].append(eval_res['precision'])
      data_dict['recall'].append(eval_res['recall'])
      data_dict['roc_auc'].append(eval_res['roc_auc'])
      data_dict['pr_auc'].append(eval_res['pr_auc'])
      data_dict['accuracy'].append(eval_res['accuracy'])

  agg_curve = agg_curves(data_dict)

  with open('{folder_prefix}/{model}_agg_{comorb_code}_{dtype}.json'.format(
    folder_prefix=ARGS.out,
    dtype=d_type,
    comorb_code=ARGS.comorb,
    model=model), 'w') as outfile:
    json.dump(agg_curve, outfile, cls=NpEnconder)

def main():
    for d_type in ['merged', 'mt', 'var']:
        for model in ['logit', 'svc', 'knn', 'dt', 'rf', 'ada', 'gbc']:
            create_agg_curves(d_type, model)

if __name__ == '__main__':
    main()
