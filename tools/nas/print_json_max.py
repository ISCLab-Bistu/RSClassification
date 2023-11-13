import os
import json

import numpy as np

import torch
import torch.nn as nn

from rmsm.models.nas.search_cnn import SearchCNN, SearchCNNController

# Load the model or state dictionary from the .pth file
checkpoint = torch.load('ovarian_cancer_darts/best.pth')
print(checkpoint)

# If the saved file is a model state dictionary
# loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
# model = SearchCNNController(input_channels=1, init_channels=16, n_classes=2, n_layers=8,
#                             loss=loss, device_ids=0)  # Define your model class
# model.load_state_dict(checkpoint)
# print(model)

# json_path = 'single_cell_transformer_backbone/show_dir/'
#
# file_list = os.listdir(json_path)
#
# f1_all = []
# best_f1 = 0
# best_file = ""
# for f in file_list:
#     with open(json_path + f, encoding='utf-8') as file:
#         result = json.load(file)
#         f1_score = result.get('f1_score')
#         print(f1_score)
#         if best_f1 < f1_score:
#             best_f1 = f1_score
#             best_file = f
#             print(f)
#         f1_all.append(f1_score)
#         file.close()
