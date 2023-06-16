import os
import json

import numpy as np

from rmsm.utils import load_json_log

# Read all results saved by cross-validation，And output the average result into onejsondocument
# path = '../data/single_cell_googlenet/show_dir/'
path = './json_data/A/show_dir/'

file_list = os.listdir(path)
file_length = len(file_list)

accuracy_top = []
precision = []
recall = []
f1_score = []
for f in file_list:
    file_path = path + f
    # Read open file
    with open(file_path, encoding='utf-8') as file:
        # Read file
        result = json.load(file)
        # Accuracy
        accuracy_top.append(result.get('accuracy1_top-1'))
        # Precision
        precision.append(result.get('precision'))
        # Recall
        recall.append(result.get('recall'))
        # f1_score
        f1_score.append(result.get('f1_score'))

print("Accuracy:{}".format(np.mean(accuracy_top)))
print("Accuracy-standard deviation:{}".format(np.std(accuracy_top)))
print("Precision:{}".format(np.mean(precision)))
print("Precision-standard deviation:{}".format(np.std(precision)))
print("Recall:{}".format(np.mean(recall)))
print("Recall-standard deviation:{}".format(np.std(recall)))
print("f1_score:{}".format(np.mean(f1_score)))
print("f1_score-standard deviation:{}".format(np.std(f1_score)))
