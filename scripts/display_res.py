'''
@author: Sybille M. Legitime

Ouput AGGREGATE  ROC and PR curves
'''

import json
import math
import argparse
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

def display_roc_curves_grid(datalist, geno_mt_config, model_name):
  cols = 3
  rows = math.ceil(len(geno_mt_config) / cols)
  line_weight=2
  label_size=12
  title_size=16

  titles_dict = {
      'geno_1000000_mt_1000000': r'$(g,m)=(\infty, \infty)$', 
      'geno_10_mt_1': r'$(g,m)=(10, 1)$', 
      'geno_10_mt_5': r'$(g,m)=(10, 5)$', 
      'geno_15_mt_1': r'$(g,m)=(15, 1)$', 
      'geno_15_mt_5': r'$(g,m)=(15, 5)$', 
      'geno_5_mt_5': r'$(g,m)=(5, 5)$'
  }

  fig, axs = plt.subplots(rows, cols, sharex='col', sharey='row', figsize=(16, rows * 4))

  for i, config in enumerate(geno_mt_config):
      i_row = i // cols
      i_col = i % cols

      res_merged = datalist[0]
      res_mt = datalist[1]
      res_var = datalist[2]

      COLOR_MERGED = '#002051'  # dark blue
      COLOR_MT = '#7f7c75' # grey
      COLOR_VAR = '#d5c164' #gold

      axs[i_row,i_col].plot(res_merged['fpr'], res_merged['tpr'], c=COLOR_MERGED, lw=line_weight, label='merged')
      axs[i_row,i_col].plot(res_mt['fpr'], res_mt['tpr'], c=COLOR_MT, lw=line_weight, label='mt')
      axs[i_row,i_col].plot(res_var['fpr'], res_var['tpr'], c=COLOR_VAR, lw=line_weight, label='var')
      axs[i_row,i_col].plot([0, 1], [0, 1], color="blue", lw=line_weight, linestyle="--")
      axs[i_row,i_col].set_xlabel("False Positive Rate", fontsize=label_size)
      axs[i_row,i_col].set_ylabel("True Positive Rate", fontsize=label_size)
      axs[i_row,i_col].text(0.65, 0.25, 'AUC = {0:0.2f}'.format(res_merged['roc_auc'][0]), color=COLOR_MERGED)
      axs[i_row,i_col].text(0.65, 0.15, 'AUC = {0:0.2f}'.format(res_mt['roc_auc'][0]), color=COLOR_MT)
      axs[i_row,i_col].text(0.65, 0.05, 'AUC = {0:0.2f}'.format(res_var['roc_auc'][0]), color=COLOR_VAR)
      axs[i_row,i_col].set_title(titles_dict[config], fontsize=title_size)

  handles, labels = axs[1,2].get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper left')

  plt.suptitle('ROC curves - {} ({})'.format(model_name.upper(), r'$c=0.8$'))
  plt.savefig(ARGS.out)

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

def main():
  datasets = []

  display_roc_curves_grid()

  # view_single(ARGS.view_single)

if __name__ == '__main__':
  main()