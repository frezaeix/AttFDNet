import os
from glob import glob
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import cv2
from scipy.io import loadmat
import cv2

class salicon(data.Dataset):
    def __init__(self,anno_dir,fix_dir,img_dir,width,height,mode='train',transform=None):
        self.anno_data = glob(os.path.join(anno_dir,mode,'*.png'))
        self.fix_dir = os.path.join(fix_dir,mode)
        #self.img_dir = img_dir
        self.img_dir = os.path.join(img_dir,mode)
        self.width = width
        self.height = height
        self.transform = transform

    def get_fixation(self,fix_data):
        fix_data = loadmat(fix_data)
        img_height, img_width = fix_data['resolution'][0]
        fixations = fix_data['gaze']
        #fixation_map = np.zeros([self.height,self.width]).astype('float32')
        fixation_map = np.zeros([2*self.height, 2*self.width]).astype('float32')
        # accumulating the fixations
        for subj_id in range(len(fixations)):
            for fix_id in range(len(fixations[subj_id][0][2])):
                x, y = fixations[subj_id][0][2][fix_id]
                #x, y = min(int(x*(self.width*1.0/img_width)),self.width-1), min(int(y*(self.height*1.0/img_height)),self.height-1)
                x, y = min(int(x * (2*self.width * 1.0 / img_width)), 2*self.width - 1), min(
                    int(y * (2*self.height * 1.0 / img_height)), 2*self.height - 1)
                fixation_map[y,x] = 1

        return torch.from_numpy(fixation_map)

    def __getitem__(self,index):
        cur_id = os.path.basename(self.anno_data[index])[:-4]
        # loading the saliency map
        cur_anno = cv2.imread(self.anno_data[index]).astype('float32')
        cur_anno = cur_anno[:,:,0]
        #cur_anno = cv2.resize(cur_anno,(self.width,self.height))
        cur_anno = cv2.resize(cur_anno, (2*self.width, 2*self.height))
        cur_anno /= cur_anno.sum()
        cur_anno = torch.from_numpy(cur_anno)
        # loading the image
        cur_img = Image.open(os.path.join(self.img_dir,cur_id+'.jpg')).convert('RGB')
        if self.transform is not None:
            cur_img = self.transform(cur_img)
        # loading the fixation map
        cur_fix = self.get_fixation(os.path.join(self.fix_dir,cur_id+'.mat'))

        return cur_img, cur_anno, cur_fix

    def __len__(self,):
        return len(self.anno_data)





