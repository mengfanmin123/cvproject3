import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #数据在网络中的维度是 batch channel width height
        #nn.Conv2d()中参数含义与顺序 输入通道，输出通道，卷积核大小，步长，padding
        self.conv1_1 = nn.Conv2d(1,8,5,2,padding=0)
        #加入BN层
        self.bn1_1 = nn.BatchNorm2d(8,affine=True)

        self.conv2_1 = nn.Conv2d(8,16,3,1,padding=0)
        self.bn2_1 = nn.BatchNorm2d(16,affine=True)

        self.conv2_2 = nn.Conv2d(16,16,3,1,padding=0)
        self.bn2_2 = nn.BatchNorm2d(16,affine=True)

        self.conv3_1 = nn.Conv2d(16,24,3,1,padding=0)
        self.bn3_1 = nn.BatchNorm2d(24,affine=True)

        self.conv3_2 = nn.Conv2d(24,24,3,1,padding=0)
        self.bn3_2 = nn.BatchNorm2d(24,affine=True)

        self.conv4_1 = nn.Conv2d(24,40,3,1,padding=1)
        self.bn4_1 = nn.BatchNorm2d(40,affine=True)

        self.conv4_2 = nn.Conv2d(40,80,3,1,padding=1)
        self.bn4_2 = nn.BatchNorm2d(80,affine=True)
        
        #PReLU 是带参数的Relu ，即在更新时，参数也会更新，因此需要分别定义PReLU
        self.relu_conv1_1 = nn.PReLU()
        self.relu_conv2_1 = nn.PReLU()
        self.relu_conv2_2 = nn.PReLU()
        self.relu_conv3_1 = nn.PReLU()
        self.relu_conv3_2 = nn.PReLU()
        self.relu_conv4_1 = nn.PReLU()
        self.relu_conv4_2 = nn.PReLU()
        self.relu_ip1 = nn.PReLU()
        self.relu_ip2 = nn.PReLU()
        #nn.AvgPool2d()中参数含义？还有什么常用的 pooling 方式？均值池化 还有max池化
        self.avgpoll = nn.AvgPool2d(2,2,ceil_mode=True)
        #nn.Linear()是什么意思？参数含义与顺序？ 全连层 输入的特征数量 输出特征数量
        self.ip1 = nn.Linear(80*4*4,128)
        self.ip2 = nn.Linear(128,128)
        self.landmarks = nn.Linear(128,42)

    def forward(self,x):

        x = self.avgpoll(self.relu_conv1_1(self.conv1_1(x)))

        x = self.relu_conv2_1(self.conv2_1(x))
        x = self.avgpoll(self.relu_conv2_2(self.conv2_2(x)))

        x = self.relu_conv3_1(self.conv3_1(x))
        x = self.avgpoll(self.relu_conv3_2(self.bn3_2(self.conv3_2(x))))     

        x = self.relu_conv4_1(self.bn4_1(self.conv4_1(x)))
        x = self.relu_conv4_2(self.bn4_2(self.conv4_2(x)))

        # view()的作用？ 是将上层的特征值 flatten 成1维
        x = x.view(-1,80*4*4)

        x = self.relu_ip1(self.ip1(x))
        x = self.relu_ip2(self.ip2(x))

        x = self.landmarks(x)

        return x


class ResNet(nn.Module):
    def __init__(self, model):
        super(ResNet, self).__init__()
        # 取掉model的后1层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.ip1 = nn.Linear(512,128)
        self.landmarks = nn.Linear(128,42)
 
    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.ip1(x)
        x = self.landmarks(x)
        return x