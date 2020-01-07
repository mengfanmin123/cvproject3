from FaceLandmarks_Cls_Network import *
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
import os
from data_mfm import *
import copy
from matplotlib import pyplot as plt
from torch.autograd import Variable

def test(args,model,trained_model,valid_loader,pts_criterion,device):
    model.load_state_dict(torch.load(os.path.join(args.save_directory,trained_model)))
    #进行正向传播 计算loss
    model.eval()
    with torch.no_grad():
        valid_loss_sum=0.0
        valid_batch_cnt=0
        for batch_idx,batch in enumerate(valid_loader):
            valid_batch_cnt +=1
            img = batch['image']
            landmarks = batch['landmarks']

            image_input = img.to(device)
            target_pts = landmarks.to(device)

            output_pts = model(image_input)

            valid_loss = pts_criterion(output_pts,target_pts) 
            valid_loss_sum += valid_loss.item()
        
        valid_mean_pts_loss = valid_loss_sum/(valid_batch_cnt * 1.0)

        print('Valid: pts_loss: {:.6f}'.format(valid_mean_pts_loss)) #Valid: pts_loss: 38.779119