
import argparse
import os
import random
import time
import warnings
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import os, sys
sys.path.append(os.getcwd())
from resnet import SupConResNet
from balanced_clustering import balanced_kmean
from kmeans_gpu import kmeans
from unbalance import IMBALANCECIFAR10, IMBALANCECIFAR100
from  loss import SupConLoss_ccl,SupConLoss_rank,SupConLoss
from utils import*

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar100', help='dataset setting')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='Rank', type=str, help='loss function for constrastive learning')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--feat_dim', default=128, type=int, help='feature dimenmion for model')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument( '--step', default=10, type=int,
                    metavar='N' ,help='steps for updating cluster')
parser.add_argument('--lr', '--learning-rate', default=0.5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--lr_decay_rate', default=0.1,type=float,
                        help='decay rate for learning rate')
parser.add_argument('--wd', '--weight-decay',default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--cluster_method', default=False, type=str, 
                    help='chose to balance cluster')
parser.add_argument('--cluster', default=10, type=int,
                    metavar='N', help='the low limit of cluster')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--temperature', default=0.1, type=float,
                    help='softmax temperature')
parser.add_argument('--cosine', default='True',
                        help='using cosine annealing')
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.batch_size > 256:
        args.warm = True
    else:
        args.warm = False
    if args.warm:
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.lr - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.lr
    num_classes = 100 if args.dataset == 'cifar100' else 10
    args.num_classes = num_classes
    model  = SupConResNet(feat_dim=args.feat_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root='./dataset/data', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=TwoCropTransform(transform_train))
    elif args.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root='./dataset/data100', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=TwoCropTransform(transform_train))
    else:
        warnings.warn('Dataset is not listed')
        return
    cls_num_list = train_dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list
    print(sum(cls_num_list))
    train_sampler = None
    train_loader_cluster = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    cluster_number= [t//max(min(cls_num_list),args.cluster) for t in cls_num_list]
    for index, value in enumerate(cluster_number):
         if value==0:
            cluster_number[index]=1
    print(cluster_number)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        if epoch < args.warm_epochs:
            criterion =SupConLoss(temperature=args.temperature).cuda()
        else:
             if args.train_rule != 'Rank':
                if epoch % args.step == 0:
                    targets=cluster(train_loader_cluster,model,cluster_number,args)
                    train_dataset.new_labels = targets 
                criterion =SupConLoss_ccl(temperature=args.temperature).cuda()
             else: 
                 if epoch % args.step == 0:
                    targets,density=cluster(train_loader_cluster,model,cluster_number,args)
                    train_dataset.new_labels = targets   
                 criterion=SupConLoss_rank(num_class=num_classes,ranking_temperature=density).cuda()
        train_loss=train(train_loader,model,criterion, optimizer, epoch,args)
        if ((epoch+1) % 100 == 0  and  100< epoch < 1000) or (epoch==999):
            save_file = ''.format(epoch=epoch) #you should add file
            torch.save(model.state_dict(),save_file)   

def cluster (train_loader_cluster,model,cluster_number,args):
    model.eval()
    features_sum = []
    for i, (input, target,cluster_target) in enumerate(train_loader_cluster):
        input = input[0].cuda()
        target =target.cuda()
        with torch.no_grad():
            features  = model(input)
            features = features.detach()
            features_sum.append(features)
    features= torch.cat(features_sum,dim=0)
    features = torch.split(features, args.cls_num_list, dim=0)
    if args.train_rule == 'Rank':
         feature_center = [torch.mean(t, dim=0) for t in features]
         feature_center = torch.cat(feature_center,axis = 0)
         feature_center=feature_center.reshape(args.num_classes,args.feat_dim)
         density = np.zeros(len(cluster_number))
         for i in range(len(cluster_number)):  
            center_distance = F.pairwise_distance(features[i], feature_center[i], p=2).mean()/np.log(len(features[i])+10) 
            density[i] = center_distance.cpu().numpy()
         density = density.clip(np.percentile(density,20),np.percentile(density,80)) 
         #density = args.temperature*np.exp(density/density.mean())
         density = args.temperature*(density/density.mean())
         for index, value in enumerate(cluster_number):
            if value==1:
                density[index] = args.temperature
    target = [[] for i in range(len(cluster_number))]
    for i in range(len(cluster_number)):  
        if cluster_number[i] >1:
          if args.cluster_method:
            cluster_ids_x, _ = balanced_kmean(X=features[i], num_clusters=cluster_number[i], distance='cosine', init='k-means++',iol=50,tol=1e-3,device=torch.device("cuda"))
          else:
            cluster_ids_x, _ = kmeans(X=features[i], num_clusters=cluster_number[i], distance='cosine', tol=1e-3, iter_limit=35, device=torch.device("cuda"))
            #run faster for cluster
          target[i]=cluster_ids_x
        else:
            target[i] = torch.zeros(1,features[i].size()[0], dtype=torch.int).squeeze(0)
    cluster_number_sum=[sum(cluster_number[:i]) for i in range(len(cluster_number))]
    for i ,k in enumerate(cluster_number_sum):
         target[i] =  torch.add(target[i], k)
    targets=torch.cat(target,dim=0)
    targets = targets.numpy().tolist()
    if args.train_rule == 'Rank':
        return targets,density
    else:
        return targets
    
    
def train(train_loader,model,criterion, optimizer, epoch, args,flag='train'):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # switch to train mode
    model.train()
    end = time.time()
    for idx, (input, target,cluster_target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = torch.cat([input[0], input[1]], dim=0)
        input = input.cuda()
        target =target.cuda()
        cluster_target = cluster_target.cuda()
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
        bsz = target.shape[0]
        # compute output
        features= model(input)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if cluster_target[0]!= -1:
            loss = criterion(features,target,cluster_target)    
        else:
            loss = criterion(features,target)
        losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
           
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        torch.cuda.empty_cache() 
    output = ('{flag} Results:  Loss {loss.avg:.5f}'
                .format(flag=flag, loss=losses))
    print(output)
    print('epoch',epoch)
    return losses.avg
def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    eta_min = lr * (args.lr_decay_rate ** 4)
    lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__ == '__main__':
    main()
