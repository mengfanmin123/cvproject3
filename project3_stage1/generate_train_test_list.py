import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from random import sample

import torch
import torch.nn.functional as F

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


# 此文件是为大家学习实现face keypoint detection所提供的参考代码。
#
# 主要是关于生成数据训练列表的参考代码
#
# 涵盖了stage1与stage3的，对于生成数据列表所需要的操作。
# 这份代码因为缺少主函数，所以【不能】直接运行，仅供参考！
# 
# 对于stage 1. 需要的操作说明文档已经十分清楚，因而对应的函数不再赘述
# 对于stage 3. 大家可能会遇到需要随机生成背景crop、或进行与人脸crop计算iou的操作。此份代码同样
#             有相对应参考代码。
#
# 希望大家仅仅是学习此份代码。并在此基础上，完成自己的代码。
# 这份代码同样还可能涵盖你可能根本用不上的东西，亦不必深究。 原则就是，挑“对自己有用”的东西学
#
# 祝大家学习顺利





folder_list = ['I', 'II']
finetune_ratio = 0.8
negsample_ratio = 0.3   # if the positive sample's iou > this ratio, we neglect it's negative samples
neg_gen_thre = 100
random_times = 3
random_border = 10
expand_ratio = 0.25

train_list_name = 'train_list.txt'
test_list_name = 'test_list.txt'

train_boarder = 112

need_record = False

train_list = 'train.txt'
test_list = 'test.txt'


def remove_invalid_image(lines):
    images = []
    for line in lines:
        name = line.split()[0]
        if os.path.isfile(name):
            images.append(line)
    return images


def load_metadata(metadata):
    tmp_lines = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for folder_name in folder_list:
        
        folder = os.path.join(current_dir+'/data', folder_name)
        metadata_file = os.path.join(folder, metadata)
        metadata_file = metadata_file.replace('\\', '/')
        
        with open(metadata_file) as f:
            lines = f.readlines()
        tmp_lines.extend(list(map((folder + '/').__add__, lines)))
    res_lines = remove_invalid_image(tmp_lines)
    return res_lines


def load_truth(lines):
    truth = {}
    for line in lines:
        line = line.strip().split()
        name = line[0]
        if name not in truth:
            truth[name] = []
        rect = list(map(int, list(map(float, line[1:5]))))
        x = list(map(float, line[5::2]))
        y = list(map(float, line[6::2]))
        landmarks = list(zip(x, y))
        truth[name].append((rect, landmarks))
    return truth

'''
1、读取图片，显示关键点和框

2、对框进行扩充，生成train test .txt

3、显示坐标框
'''

def image_expend_roi(img_w,img_h,bbox_x1,bbox_y1,bbox_x2,bbox_y2,scale):
    bbox_w = bbox_x2 - bbox_x1
    bbox_h = bbox_y2 - bbox_y1

    ori_x = bbox_x1 - int(scale*bbox_w)
    ori_y = bbox_y1 - int(scale*bbox_h)

    if ori_x < 0:
        ori_x = 0
    if ori_y < 0:
        ori_y = 0

    ori_x_end = bbox_x2 + int(scale*bbox_w)
    ori_y_end = bbox_y2 + int(scale*bbox_h)

    if ori_x_end >=img_w:
        ori_x_end = img_w-1
    
    if ori_y_end >=img_h:
        ori_y_end =  img_h -1

    return ori_x,ori_y,ori_x_end,ori_y_end

#显示出bbox及landmarks
def show_keypoinit_bbox():
    lines=load_metadata('label.txt')   
    for line in lines:
        mdata=line.strip().split()
        img=cv2.imread(mdata[0],1)
        rect = list(map(int,list(map(float,mdata[1:5]))))
        cv2.rectangle(img, (rect[0],rect[1]), (rect[2],rect[3]), (0,255,0), 2)
        x1,y1,x2,y2=image_expend_roi(img.shape[1],img.shape[0],rect[0],rect[1], rect[2],rect[3],0.25)
        cv2.rectangle(img, (x1,y1),(x2,y2), (0,255,255), 2)
        for i in range(21):
            x =int(float(mdata[5+2*i]))
            y =int(float(mdata[5+2*i+1]))
            cv2.circle(img, (x,y), 2, (0,0,255), -1,4)
        cv2.imshow('img',img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_all_labeldata(lines):
    all_data={}
    for line in lines:
        mdata=line.strip().split()
        name = mdata[0]
        if name not in all_data:
            all_data[name] =[]
        rect = list(map(int,list(map(float,mdata[1:5]))))
        x = list(map(float,mdata[5::2]))
        y = list(map(float,mdata[6::2]))
        marks= list(zip(x,y))
        all_data[name].append((rect,marks))
    return all_data


def save_to_file(lines,fo):
    ss =' '
    _data=load_all_labeldata(lines)
    for name,value in _data.items():
        for v in value:
            rect = v[0]
            marks = v[1]
            img=cv2.imread(name,1)
            x1,y1,x2,y2=image_expend_roi(img.shape[1],img.shape[0],rect[0],rect[1],rect[2],rect[3],0.25)
            new_roi = [x1,y1,x2,y2]
            new_marks = np.array(marks) - np.array([x1,y1])
            fo.write(name + ' '+ss.join(map(str,new_roi))+' '+ss.join(map(str,new_marks.flatten()))+'\n')


def generate_train_test_file():
    lines=load_metadata('label.txt')
    random.shuffle(lines)
    fo_train = open("train.txt", "w")
    fo_test = open("test.txt", "w")
    len_train = int(len(lines)*0.9)
    save_to_file(lines[:len_train],fo_train)
    save_to_file(lines[len_train:],fo_test)
    fo_train.close()
    fo_test.close()

if __name__ == '__main__':
    #show_keypoinit_bbox()
    generate_train_test_file()