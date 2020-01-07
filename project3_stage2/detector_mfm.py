
from FaceLandmarks_Network import *
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
import torchvision.models as models
from test import *
from predict import *
def adjust_learning_rate(args,optimizer,epoch):

    lr = args.lr * (0.1 ** (epoch // 40))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args,train_loader,valid_loader,model,criterion,optimizer,device):
    #判断是否需要保存model
    if args.save_model :
        f = open('train.log','w+')
        if not os.path.exists(args.save_directory):
            os.mkdir(args.save_directory)
    epoch = args.epochs
    pts_criterion = criterion

    train_losses = []
    valid_losses = []
    smaller_loss = 1e6

    for epochid in range(epoch):
        model.train()
        #adjust_learning_rate(args,optimizer,epochid)
        train_loss_sum=0.0
        train_batch_cnt=0
        for batch_idx,batch in enumerate(train_loader):
            train_batch_cnt +=1
            image = batch['image']
            landmarks = batch['landmarks']

            image_input = image.to(device)
            target_pts = landmarks.to(device)
            optimizer.zero_grad()
            
            output_pts=model(image_input)

            loss = pts_criterion(output_pts,target_pts)
            loss.backward()
            optimizer.step()
            train_loss_sum +=loss.item()
            if batch_idx % args.log_interval == 0:

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t pts_loss: {:.6f}'.format(
                        epochid,
                        batch_idx * len(image),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item()
                    )
                )
        train_mean_loss=train_loss_sum/(train_batch_cnt*1.0)
        print('Train: pts_loss: {:.6f}'.format(train_mean_loss))
        if args.save_model :
            f.write('Train: pts_loss: {:.6f}\n'.format(train_mean_loss))
        train_losses.append(train_mean_loss)

        # validate 
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

            print('Valid: pts_loss: {:.6f}'.format(valid_mean_pts_loss))
            if args.save_model :
                f.write('Valid: pts_loss: {:.6f}\n'.format(valid_mean_pts_loss))
            valid_losses.append(valid_mean_pts_loss)

            if valid_mean_pts_loss < smaller_loss:
                smaller_loss = valid_mean_pts_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best model: {:.6f}'.format(smaller_loss))

        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
     #save model
    if args.save_model:
        f.close()
        model.load_state_dict(best_model_wts)
        saved_model_name = os.path.join(args.save_directory, 'detector_mfm'+ '.pt')
        torch.save(model.state_dict(), saved_model_name)
    
    return train_losses,valid_losses
        

def do_main():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',				
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',		
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='Train',   # Train/train, Predict/predict, Finetune/finetune
                        help='training, predicting or finetuning')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")

    print('loading data set......')
    train_set,test_set = get_train_test_set()
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set,batch_size=args.test_batch_size)
    
    print('building model......')
    
    #使用resnet
    resnet = models.resnet18(pretrained=False)
    model = ResNet(resnet)
    model = model.to(device)
    
    #model = Net().to(device)
    criterion = nn.MSELoss()
    #optimizer = optim.SGD(model.parameters(),lr = args.lr,momentum=args.momentum)
    optimizer=optim.Adam(model.parameters(),lr=args.lr,betas=(0.9,0.99))
    if args.phase == 'train' or args.phase=='Train':
        print('start train......')
        train_loss,valid_loss=train(args,train_loader,valid_loader,model,criterion,optimizer,device)
        x = np.linspace(0, args.epochs, args.epochs)
        y_train = np.array(train_loss)
        y_valid = np.array(valid_loss)

        plt.plot(x, y_train, label="Train")
        plt.plot(x, y_valid, label="Valid")
        plt.legend()
        file_name = './result_batch-size_'+str(args.batch_size)+'_lr_'+str(args.lr)+'_momentum_'+str(args.momentum)+'.png'
        plt.savefig(file_name)#保存图片
        #python detector_mfm.py --lr 0.000001 --momentum 0.9 --batch-size 128 --save-model
    elif args.phase == 'test' or args.phase=='Test':
        print('start test .......')
        test(args,model,'detector_mfm.pt',valid_loader,criterion,device)
        

    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Finetune')
        #load model
        model.load_state_dict(torch.load(os.path.join('trained_models', 'detector_mfm.pt')))    
        #固定某些层     
        for name, param in model.named_parameters():
            if 'conv1_1' in name or 'conv2_1' in name or 'conv2_2' in name or 'conv3_1' in name or 'conv3_2' in name:
                param.requires_grad = False      
        
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr,momentum=args.momentum)
        train_loss,valid_loss=train(args,train_loader,valid_loader,model,criterion,optimizer,device)
        # how to do finetune?
        #python detector_mfm.py --phase finetune --save-directory finetuen_models --lr 0.000001 --momentum 0.7 --batch-size 128 --save-model --epochs 100
    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')
        # how to do predict?
        predict(args, 'detector_mfm.pt', model, valid_loader)

if __name__ == "__main__":
    do_main()