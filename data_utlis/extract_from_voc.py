import os
import os.path
import shutil
from data.config import VOCroot

fileDir_ann = VOCroot + 'VOC2007/Annotations/'
fileDir_img = VOCroot + 'VOC2007/JPEGImages/'
obj_name = 'VOC2007-IL-target_11to20/'
saveDir_img = VOCroot + obj_name + 'JPEGImages/'
print(obj_name)

if not os.path.exists(saveDir_img):
    os.mkdir(saveDir_img)

names = locals()


#selected_class = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow')
selected_class = ('diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
#selected_class = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle')
#selected_class = ('bus', 'car', 'cat', 'chair', 'cow')
#selected_class = ('diningtable', 'dog', 'horse', 'motorbike', 'person')
#selected_class = ('pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

#selected_class = ['tvmonitor']



class_line = []
for class_value in selected_class:
    class_line.append('\t\t<name>' + class_value + '</name>\n')

for files in os.walk(fileDir_ann):
    for file in files[2]:

        print(file + "-->start!")

        saveDir_ann = VOCroot + obj_name + 'Annotations/'

        if not os.path.exists(saveDir_ann):
            os.mkdir(saveDir_ann)

        fp = open(fileDir_ann + '/' + file)
        saveDir_ann = saveDir_ann + file
        fp_w = open(saveDir_ann, 'w')
        classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor']

        lines = fp.readlines()

        ind_start = []
        ind_end = []
        lines_id_start = lines[:]
        lines_id_end = lines[:]

        while "\t<object>\n" in lines_id_start:
            a = lines_id_start.index("\t<object>\n")
            ind_start.append(a)
            lines_id_start[a] = "delete"

        while "\t</object>\n" in lines_id_end:
            b = lines_id_end.index("\t</object>\n")
            ind_end.append(b)
            lines_id_end[b] = "delete"

        i = 0
        for k in range(0, len(ind_start)):
            for j in range(0, len(classes)):
                if classes[j] in lines[ind_start[i] + 1]:
                    a = ind_start[i]
                    '''
                    names['block%d' % k] = [lines[a], lines[a + 1], \
                                            lines[a + 2], lines[a + 3], lines[a + 4], \
                                            lines[a + 5], lines[a + 6], lines[a + 7], \
                                            lines[a + 8], lines[a + 9], lines[a + 10], \
                                            lines[ind_end[i]]]
                    '''
                    b = ind_end[i]
                    names['block%d' % k] = lines[a:b+1]
                    break
            i += 1

        string_start = lines[0:ind_start[0]]
        string_end = [lines[len(lines) - 1]]

        a = 0
        for k in range(0, len(ind_start)):
            for class_value in class_line:
                if class_value in names['block%d' % k]:
                    a += 1
                    string_start += names['block%d' % k]

        string_start += string_end
        for c in range(0, len(string_start)):
            fp_w.write(string_start[c])
        fp_w.close()

        if a == 0:
            os.remove(saveDir_ann)
        else:
            name_img = fileDir_img + os.path.splitext(file)[0] + ".jpg"
            shutil.copy(name_img, saveDir_img)
        fp.close()
