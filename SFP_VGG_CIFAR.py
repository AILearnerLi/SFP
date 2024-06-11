import math
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ptflops import get_model_complexity_info
 

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
# test()

model = VGG('VGG16').cuda()
model.load_state_dict(torch.load('./para/VGG16_164_CIFAR10'))
print(model)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./dataset/cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./dataset/cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
criterion = nn.CrossEntropyLoss()


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))




layer_id = 1

cfg = []

cfg_mask = []
purn_rate=0.995
prune_rate1=1-purn_rate
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG16jz': [64, int(64*0.4), 'M', 128, 128, 'M', int(256*0.6), 256, int(256*0.6), 'M', int(512*prune_rate1), int(512*prune_rate1), int(512*prune_rate1), 'M', int(512*prune_rate1), int(512*prune_rate1), int(512*prune_rate1), 'M'],
}

for m in model.modules():
    if isinstance(m, nn.Conv2d):
        if m.kernel_size == (1,1):
            continue
        elif(layer_id ==2 or layer_id ==5 or layer_id ==7):
            if(layer_id ==2):
                prune_prob_stage = 0.5
            else:
                prune_prob_stage = 0
            out_channels = m.weight.data.shape[0]
            feauture_width=m.weight.data.shape[3]
            weight_copy = m.weight.data.clone().cpu().numpy()
            weight_c =  m.weight.data.abs().clone().cpu().numpy()
            sums=[0 for i in range(out_channels)]
            for i in range(out_channels):
                L1_norm = weight_c[i]
                sums[i]=np.sum(L1_norm,axis=(0,1,2))
            in_channel=weight_copy.shape[1]
            cn=int((1-prune_prob_stage)*out_channels)
            arr1=KMeans(cn).fit_predict(weight_copy,(out_channels,in_channel*weight_copy.shape[2]*weight_copy.shape[2]))
            mask=torch.zeros(out_channels)
            for j in range(cn):
                maxs=0
                l=0
                for i in range(out_channels):
                    if(arr1[i]==j):
                        if(l==0):
                            maxs=i
                            mask[maxs]=1 
                            l=1
                        elif(sums[i]>sums[maxs]):
                            mask[maxs]=0
                            maxs=i
                            mask[maxs]=1 
            #mask[arg_max_rev.tolist()] = 1
            cfg_mask.append(mask)
            #cfg.append(num_keep)
            layer_id += 1
        elif(layer_id <7):
            out_channels = m.weight.data.shape[0]
            mask=torch.ones(out_channels)
            cfg_mask.append(mask)
            layer_id += 1
        else:
            prune_prob_stage=purn_rate
            out_channels = m.weight.data.shape[0]
            feauture_width=m.weight.data.shape[3]
            weight_copy = m.weight.data.clone().cpu().numpy()
            weight_c =  m.weight.data.abs().clone().cpu().numpy()
            sums=[0 for i in range(out_channels)]
            for i in range(out_channels):
                L1_norm = weight_c[i]
                sums[i]=np.sum(L1_norm,axis=(0,1,2))
            in_channel=weight_copy.shape[1]
            cn=int((1-prune_prob_stage)*out_channels)
            arr1=KMeans(cn).fit_predict(weight_copy,(out_channels,in_channel*weight_copy.shape[2]*weight_copy.shape[2]))
            mask=torch.zeros(out_channels)
            for j in range(cn):
                maxs=0
                l=0
                for i in range(out_channels):
                    if(arr1[i]==j):
                        if(l==0):
                            maxs=i
                            mask[maxs]=1 
                            l=1
                        elif(sums[i]>sums[maxs]):
                            mask[maxs]=0
                            maxs=i
                            mask[maxs]=1 
            #mask[arg_max_rev.tolist()] = 1
            cfg_mask.append(mask)
            #cfg.append(num_keep)
            layer_id += 1
newmodel = VGG('VGG16jz').cuda()

start_mask = torch.ones(3)
layer_id_in_cfg = 0
conv_count = 1
bias_count = 1
b_layer_id_in_cfg = 0

for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.Conv2d):
        if conv_count == 1:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            layer_id_in_cfg += 1
            m1.bias.data = m0.bias.data.clone()
            print('Conv\t no prun'
            )
            continue
        if conv_count  == 2:
            mask = cfg_mask[layer_id_in_cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[idx.tolist(), :, :, :].clone()
            m1.weight.data = w.clone()
            b = m0.bias.data.clone()
            b =b[idx.tolist()].clone()
            m1.bias.data = b.clone()
            layer_id_in_cfg += 1
            conv_count += 1
            print('Conv\t total channel: {:d} \t remaining channel: {:d}'.
            format( mask.shape[0], int(torch.sum(mask))))
            start_mask = mask
            continue
        if conv_count ==13:
            mask = cfg_mask[layer_id_in_cfg]
            idx = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[:, idx.tolist(), :, :].clone()
            m1.weight.data = w.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.weight.data = w.clone()
            b = m0.bias.data.clone()
            b =b[idx.tolist()].clone()
            layer_id_in_cfg += 1
            conv_count += 1
            print('Conv\t no prun')
            start_mask = mask
        else:
            mask = cfg_mask[layer_id_in_cfg]
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
                idx1 = np.resize(idx1, (1,))
            idx=list(set(idx0) & set(idx1))
            w = m0.weight.data.clone()
            w =w[idx.tolist(), :, :, :].clone()
            m1.weight.data = w.clone()
            b = m0.bias.data.clone()
            b =b[idx1.tolist()].clone()
            m1.bias.data = b.clone()
            print('Conv\t total channel: {:d} \t remaining channel: {:d}'.
            format( mask.shape[0], int(torch.sum(mask))))
            conv_count += 1
            layer_id_in_cfg += 1
            start_mask = mask
            continue
    elif isinstance(m0, nn.BatchNorm2d):
        assert isinstance(m1, nn.BatchNorm2d), "There should not be bn layer here."
        if conv_count !=2 and conv_count !=14 :
            mask = cfg_mask[layer_id_in_cfg-1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            m1.weight.data = m0.weight.data[idx.tolist()].clone()
            m1.bias.data = m0.bias.data[idx.tolist()].clone()
            m1.running_mean = m0.running_mean[idx.tolist()].clone()
            m1.running_var = m0.running_var[idx.tolist()].clone()
            print('BN\t total channel: {:d} \t remaining channel: {:d}'.
            format( mask.shape[0], int(torch.sum(mask))))
        else:
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
            print('BN\t no prun')
    elif isinstance(m0, nn.Linear):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        print('Lin \t no prun')
def newtest(epoch):
    global best_acc
    newmodel.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = newmodel(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
        c1=100.*correct/total
        c1=str(c1)
        with open("SVD_JZ_7-.txt","a+") as f:
            f.write('newmodel acc(before train):')
            f.write(c1)
            f.write('\n')
def newtrain(epoch):
    print('\nEpoch: %d' % epoch)
    newmodel.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = newmodel(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

optimizer = optim.SGD(newmodel.parameters(), lr=0.0001,
                    momentum=0.9, weight_decay=5e-4)
#lr_scheduler = optim.lr_scheduler.Cosin
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
for epoch in range(0, 50):
    criterion = nn.CrossEntropyLoss()
    newtrain(epoch)
    lr_scheduler.step()
    newtest(epoch)
