# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time
from randomaug import RandAugment

from utils import *

import timm

from tensorboardX import SummaryWriter

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='use randomaug')
parser.add_argument('--amp', action='store_true', help='enable AMP training')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', type=str, default='vit-b')
parser.add_argument('--bs', type=int, default='256')
parser.add_argument('--size', type=int, default="224")
parser.add_argument('--n_epochs', type=int, default='50')
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int)
parser.add_argument('--cos', action='store_false', help='Train with cosine annealing scheduling')

parser.add_argument('--seed', default=1)

args = parser.parse_args()

# Tensorboard
tb_writer = SummaryWriter(log_dir=f'./runs/{args.net}')

bs = int(args.bs)
imsize = int(args.size)

use_amp = args.amp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(args.size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.Resize(args.size),
    transforms.ToTensor(),
])

# Add RandAugment with N, M(hyperparameter)
if args.aug:  
    N = 2
    M = 14
    transform_train.transforms.insert(0, RandAugment(N, M))

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.net == 'resnet18':
    net = timm.create_model('resnet18', pretrained=True, num_classes=10)
elif args.net == 'resnet34':
    net = timm.create_model('resnet34', pretrained=True, num_classes=10)
elif args.net == 'resnet50':
    net = timm.create_model('resnet50', pretrained=True, num_classes=10)
elif args.net == 'wrn50-2':
    net = timm.create_model('wide_resnet50_2', pretrained=True, num_classes=10)
elif args.net == 'wrn101-2':
    net = timm.create_model('wide_resnet101_2', pretrained=True, num_classes=10)
elif args.net == 'inc_v3':
    net = timm.create_model('inception_v3', pretrained=True, num_classes=10)
elif args.net == 'inc_v4':
    net = timm.create_model('inception_v4', pretrained=True, num_classes=10)
elif args.net == 'bit50-3':
    net = timm.create_model('resnetv2_50x3_bitm', pretrained=True, num_classes=10)
elif args.net == 'bit101-3':
    net = timm.create_model('resnetv2_101x3_bitm', pretrained=True, num_classes=10)
elif args.net == 'bit152-4':
    net = timm.create_model('resnetv2_152x4_bitm', pretrained=True, num_classes=10)
elif args.net == 'vit-b':
    net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
elif args.net == 'vit-s':
    net = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=10)
elif args.net == 'vit-t':
    net = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10)
elif args.net == 'deit-b':
    net = timm.create_model('deit_base_patch16_224', pretrained=True, num_classes=10)
elif args.net == 'deit-s':
    net = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=10)
elif args.net == 'deit-t':
    net = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=10)
elif args.net == 'swin-b':
    net = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=10)
elif args.net == 'swin-s':
    net = timm.create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=10)
elif args.net == 'swin-t':
    net = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=10)
else:
    raise 'no matched model'

if device == 'cuda':
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True
    torch.manual_seed(args.seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(args.seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置
    np.random.seed(args.seed)

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.AdamW(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine or reduce LR on Plateau scheduling
if not args.cos:
    from torch.optim import lr_scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

iter = 0


def train(epoch):
    global iter
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        # with torch.cuda.amp.autocast(enabled=use_amp):
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (iter + 1) % 10 == 0:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            tb_writer.add_scalar('Train/Loss', loss.item(), iter)
            tb_writer.add_scalar('Train/Acc', correct/total, iter)
        iter = iter + 1

    return train_loss/(batch_idx+1)


##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Update scheduler
    if not args.cos:
        scheduler.step(test_loss)
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"state_dict": net.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict()}
        save_path = os.path.join('checkpoint', args.net)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        torch.save(state, os.path.join(save_path, f'{args.net}_best.pth.tar'))
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}.txt', 'a') as appender:
        appender.write(content + "\n")

    tb_writer.add_scalar('Test/Loss', loss.item(), epoch)
    tb_writer.add_scalar('Test/Acc', acc/100, epoch)

    return test_loss, acc


if __name__ == '__main__':
    list_loss = []
    list_acc = []

    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        trainloss = train(epoch)
        val_loss, acc = test(epoch)

        if args.cos:
            scheduler.step(epoch-1)

        list_loss.append(val_loss)
        list_acc.append(acc)

        # Write out csv..
        with open(f'log/log_{args.net}.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(list_loss)
            writer.writerow(list_acc)
        print(list_loss)

    print('finished training!')
    
