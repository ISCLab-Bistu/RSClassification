# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import json
import os

save_flg = True

path_dir = './json_data/A/cell/'
file_list = os.listdir(path_dir)
index = 3

pnas_class = ['HN', 'CVB']
ovarian_class = ['Healthy', 'Ovarian']
single_class = ['CLP', 'CMP', 'HSC', 'MPP1', 'MPP2', 'MPP3']
cell_class = ['MC', 'NHPM', 'NSF', 'PMC', 'TAF']
covid_class = ['COVID', 'Health', 'Suspected']

class_list = [pnas_class, ovarian_class, single_class, cell_class, covid_class]
str_list = ['RNA virus', 'Ovarian cancer', 'Hematopoietic cell population',
            'Cancer cell culture medium', 'COVID-19']


arr_length = len(class_list[index])
confusion_all = np.zeros((arr_length, arr_length), dtype=int)
for f in file_list:
    file_path = path_dir + f
    # 
    with open(file_path, encoding='utf-8') as file:
        result = json.load(file)
        # Getting the confusion matrix
        confusion = result.get('confusion')
        confusion = np.array(confusion)
        confusion_all += confusion

        file.close()

print(confusion_all)

plt.figure(figsize=(8, 6), dpi=500)  # set the image size
# Set the global font
# In this example, the axis scales and legends are in ['TimesNewRoman']
plt.rc('font', family='Times New Roman')

# 1. Heat map, followed by the specified color block, cmap can set other different colors
plt.imshow(confusion_all, cmap=plt.cm.Blues)
plt.colorbar()  # colorbar

# 2. Set the axis to display the list
indices = range(len(confusion_all))
classes = class_list[index]

# 3. The first is the iteration object，Represents the display order of coordinates，The second parameter is the axis
# display list
plt.xticks(indices, classes, fontsize=16)  # Set the horizontal direction，rotation=4545
plt.yticks(indices, classes, fontsize=16)

# 4. Set axis titles and fonts
plt.xlabel('Predicted label', fontsize=18)
plt.ylabel('True label', fontsize=18)
plt.title('Tag: ' + str_list[index], fontsize=18)  # 、

# 5. display
normalize = False
fmt = '.2f' if normalize else 'd'
thresh = confusion_all.max() / 2.

for i in range(len(confusion_all)):
    for j in range(len(confusion_all[i])):
        plt.text(j, i, format(confusion_all[i][j], fmt),
                 fontsize=18,
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if confusion_all[i, j] > thresh else "black")

# 6. save tif
# plt.savefig("./A/" + str_list[index] + ".tif")

# 7. show
plt.show()
