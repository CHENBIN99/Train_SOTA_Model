# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import argparse
import csv

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter

from dataloader import get_dataloader
from model import get_model
from utils import *

iter = 0


# parsers
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4?
    parser.add_argument('--opt', default="adam")
    parser.add_argument('--aug', action='store_true', help='use randomaug')
    parser.add_argument('--amp', action='store_true', help='enable AMP training')
    parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
    parser.add_argument('--net', type=str, default='vit-b')
    parser.add_argument('--bs', type=int, default='256')
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--size', type=int, default="224")
    parser.add_argument('--n_epochs', type=int, default='50')
    parser.add_argument('--cos', action='store_false', help='Train with cosine annealing scheduling')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'tinyimagenet'])
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--seed', default=1)
    args = parser.parse_args()
    return args


def train(epoch):
    global iter
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
        for batch_idx, (inputs, targets) in enumerate(valid_dataloder):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valid_dataloder), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Update scheduler
    if not args.cos:
        scheduler.step(test_loss)
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"state_dict": net.state_dict(),
                 "optimizer": optimizer.state_dict()}
        save_path = os.path.join('checkpoint', args.dataset, args.net)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        torch.save(state, os.path.join(save_path, f'{args.net}_best.pth.tar'))
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.dataset}_{args.net}.txt', 'a') as appender:
        appender.write(content + "\n")

    tb_writer.add_scalar('Test/Loss', loss.item(), epoch)
    tb_writer.add_scalar('Test/Acc', acc/100, epoch)

    return test_loss, acc


if __name__ == '__main__':
    args = get_parser()
    setattr(args, 'root_path', get_project_path())

    list_loss = []
    list_acc = []

    # Tensorboard
    tb_writer = SummaryWriter(log_dir=f'./runs/{args.dataset}/{args.net}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    train_dataloader, valid_dataloder, num_classes = get_dataloader.get_dataloader(args)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    net = get_model.get_model(args, num_classes)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)  # make parallel
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

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3 * 1e-5,
                                                   factor=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        trainloss = train(epoch)
        val_loss, acc = test(epoch)

        if args.cos:
            scheduler.step(epoch-1)

        list_loss.append(val_loss)
        list_acc.append(acc)

        # Write out csv..
        with open(f'log/log_{args.dataset}_{args.net}.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(list_loss)
            writer.writerow(list_acc)
        print(list_loss)

    print('finished training!')
    
