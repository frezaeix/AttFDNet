import os
import os.path
import shutil
from data.config import VOCroot

doc_name = 'VOC2007-IL-source_1to10/'
Dir_img = VOCroot + doc_name + 'JPEGImages/'
Dir_ann = VOCroot + doc_name + 'Annotations/'
maindir = VOCroot + doc_name + 'ImageSets/Main/'
origin_maindir = VOCroot + doc_name + 'ImageSets/Origin_Main/'

print(doc_name)

files = os.listdir(Dir_img)
image_list = []
for i in range(len(files)):
    file_prefix = files[i][:-4]
    image_list.append(file_prefix)
image_set = set(image_list)


# split the train set
train_txt_list = []
with open(origin_maindir + 'train.txt') as f:
    for line in f.readlines():
        line = line.rstrip('\n')
        train_txt_list.append(line)
train_txt_set = set(train_txt_list)

intersection_image_train = image_set & train_txt_set
intersection_image_train_list = list(intersection_image_train)
intersection_image_train_list.sort()

with open(maindir + "train.txt", "w") as f:
    for line in intersection_image_train_list:
        f.write(line + "\n")


# split the trainval set
train_txt_list = []
with open(origin_maindir + 'trainval.txt') as f:
    for line in f.readlines():
        line = line.rstrip('\n')
        train_txt_list.append(line)
train_txt_set = set(train_txt_list)

intersection_image_train = image_set & train_txt_set
intersection_image_train_list = list(intersection_image_train)
intersection_image_train_list.sort()

with open(maindir + "trainval.txt", "w") as f:
    for line in intersection_image_train_list:
        f.write(line + "\n")


# split the test set
train_txt_list = []
with open(origin_maindir + 'test.txt') as f:
    for line in f.readlines():
        line = line.rstrip('\n')
        train_txt_list.append(line)
train_txt_set = set(train_txt_list)

intersection_image_train = image_set & train_txt_set
intersection_image_train_list = list(intersection_image_train)
intersection_image_train_list.sort()

with open(maindir + "test.txt", "w") as f:
    for line in intersection_image_train_list:
        f.write(line + "\n")

print("spliting done")
