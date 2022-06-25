import os
import torch
import torchvision
import torchvision.transforms as transforms

from dataloader import tinyimagenet_dataloader

from randomaug import RandAugment

from utils import *

def get_transform(args):
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
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

    elif args.dataset == 'tinyimagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.Resize(args.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(args.size),
            transforms.ToTensor(),
        ])

    return transform_train, transform_test


def get_dataloader(args):
    transform_train, transform_test = get_transform(args)

    # Add RandAugment with N, M(hyperparameter)
    if args.aug:
        N = 2
        M = 14
        transform_train.transforms.insert(0, RandAugment(N, M))

    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.root_path, 'data', 'cifar10'), train=True,
                                                download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True,
                                                  num_workers=args.num_worker, pin_memory=True)

        testset = torchvision.datasets.CIFAR10(root=os.path.join(args.root_path, 'data', 'cifar10'), train=False,
                                               download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                                 num_workers=args.num_worker, pin_memory=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(args.root_path, 'data', 'cifar100'),
                                                 train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True,
                                                  num_workers=args.num_worker, pin_memory=True)

        testset = torchvision.datasets.CIFAR100(root=os.path.join(args.root_path, 'data', 'cifar100'), train=False,
                                                download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                                 num_workers=args.num_worker, pin_memory=True)
        num_classes = 100
    elif args.dataset == 'tinyimagenet':
        if not os.path.exists(os.path.join(args.root_path, './data', 'tiny-imagenet-200')):
            download_tinyimagenet(args)
        # trainset = tinyimagenet_dataloader.TinyImageNet(root=os.path.join(args.root_path, 'data', 'tiny-imagenet-200'),
        #                                                 train=True, transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=False,
        #                                           num_workers=args.num_worker, pin_memory=True)
        # testset = tinyimagenet_dataloader.TinyImageNet(root=os.path.join(args.root_path, 'data', 'tiny-imagenet-200'),
        #                                                train=False, transform=transform_train)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False,
        #                                          num_workers=args.num_worker, pin_memory=True)
        trainloader, testloader = tinyimagenet_dataloader.get_tiny(args)
        num_classes = 200



    else:
        raise 'no match dataset'

    return trainloader, testloader, num_classes
