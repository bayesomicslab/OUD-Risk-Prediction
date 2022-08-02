import json
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CLI argument parser
parser = argparse.ArgumentParser(description='Options to parameterize the display result script.')
parser.add_argument('--evaluation_res', type=str, help='Path to dataset folder')
parser.add_argument('--view_single', type=str, help='Type of model to view e.g \'svc_train\'')
parser.add_argument('--out', type=str, help='Path to ouput file')
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

def output_roc_pr_curves(datalist, model_name):
  _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
  merged =  datalist[0]
  mt = datalist[1]
  var = datalist[2]
  # plotting ROC curve 
  ax1.plot(merged['fpr'], merged['tpr'], lw=2, label='merged ROC curve (area = {0:0.2f})'.format(merged['roc_auc']))
  ax1.plot(mt['fpr'], mt['tpr'], lw=2, label='mob_trace ROC curve (area = {0:0.2f})'.format(mt['roc_auc']))
  ax1.plot(var['fpr'], var['tpr'], lw=2, label='variant ROC curve (area = {0:0.2f})'.format(var['roc_auc']))
  ax1.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
  ax1.set_xlabel("False Positive Rate")
  ax1.set_ylabel("True Positive Rate")
  ax1.legend(loc="best")
  ax1.set_title("ROC curve - {}".format(model_name))

  # plot the model Precision-Recall curve
  ax2.plot(merged['recall'], merged['precision'], lw=2, label='merged PR curve (area = {0:0.2f})'.format(merged['pr_auc']))
  ax2.plot(mt['recall'], mt['precision'], lw=2, label='mob_trace PR curve (area = {0:0.2f})'.format(mt['pr_auc']))
  ax2.plot(var['recall'], var['precision'], lw=2, label='variant PR curve (area = {0:0.2f})'.format(var['pr_auc']))
  ax2.set_xlabel("Recall")
  ax2.set_ylabel("Precision")
  ax2.legend(loc="best")
  ax2.set_title("Precision-Recall curve - {}".format(model_name))

  plt.savefig(ARGS.out)

def agg_curves(data_dict):
  metrics_array = ['fpr', 'tpr', 'precision', 'recall']
  for metric in metrics_array:
    length = max(map(len, data_dict[metric]))

    data_dict[metric] = np.concatenate([np.array([np.pad(a, (0, length - len(a)), 'maximum')]) for a in data_dict[metric]], axis=0)
    data_dict[metric] = np.mean(data_dict[metric], axis=0)

  data_dict['roc_auc'] = np.mean(np.array(data_dict['roc_auc']))
  data_dict['pr_auc'] = np.mean(np.array(data_dict['pr_auc']))

  return data_dict

def view_single(model_name):
  substr = model_name.split('_')
  evals = [
    '{}/{}_agg_07_merged.json'.format(substr[0], substr[0]), 
    '{}/{}_agg_07_mt.json'.format(substr[0], substr[0]), 
    '{}/{}_agg_07_var.json'.format(substr[0], substr[0])
  ]
  evals_2 = [
    '{}/merged/eval_output_merged_{}_test_0.json'.format(substr[0], substr[0]),
    '{}/mt/eval_output_mt_{}_test_0.json'.format(substr[0], substr[0]),
    '{}/var/eval_output_var_{}_test_0.json'.format(substr[0], substr[0]),
  ]
  data_list = []

  for eval in evals:
    with open('{}/{}'.format(ARGS.evaluation_res, eval)) as json_file:
      data_list.append(json.load(json_file))
  
  output_roc_pr_curves(data_list, model_name)

def create_agg_curves():
  data_dict = {
      'fpr': [],
      'tpr': [],
      'precision': [],  
      'recall': [],
      'roc_auc': [],
      'pr_auc': []
    }

  for name in glob.glob('{}/logit/merged/*'.format(ARGS.evaluation_res)):
    with open(name, 'r') as json_file:
      eval_res = json.load(json_file)
      data_dict['fpr'].append(eval_res['fpr']['1'])
      data_dict['tpr'].append(eval_res['tpr']['1'])
      data_dict['precision'].append(eval_res['precision']['1'])
      data_dict['recall'].append(eval_res['recall']['1'])
      data_dict['roc_auc'].append(eval_res['roc_auc']['pos'])
      data_dict['pr_auc'].append(eval_res['pr_auc']['pos'])

  agg_curve = agg_curves(data_dict)

  with open('./test_res/svc/svc_agg_07_var.json', 'w') as outfile:
    json.dump(agg_curve, outfile, cls=NpEnconder)

def main():
  create_agg_curves()

  # view_single(ARGS.view_single)

if __name__ == '__main__':
  main()