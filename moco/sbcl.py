import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import moco
from kmeans_pytorch import kmeans
from imagenet_lt_loader import ImageNetLT_moco
from kcl import KCL
from utils import*
from loss import*

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--cluster', default=20, type=int,
                    help='contorl cluster number')
parser.add_argument('--step', default=5, type=int,
                    help='step for updating cluster')
parser.add_argument('--train_rule', default='SCL', type=str, help='strategy for train loader')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', default='True',
                    help='use mlp head')
parser.add_argument('--aug-plus', default='True',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', default='True',
                    help='use cosine lr schedule')

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

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print('distributed',args.distributed)
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    args.data = 'autodl-tmp/imagenet'
    traindir = os.path.join(args.data, 'train')
    txt_train = f'moco/ImageNet_LT_train.txt'
    txt_test = f'moco/ImageNet_LT_test.txt'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    transform_train = [transforms.Compose(augmentation), transforms.Compose(augmentation)]
    augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]   
    train_dataset = ImageNetLT_moco(
        root=args.data,
        txt=txt_train,
        transform=transform_train)
    args.num_class = 1000
    args.cls_num_list = train_dataset.cls_num_list
    cluster_number= [t//max(min(args.cls_num_list),args.cluster) for t in args.cls_num_list]
    for index, value in enumerate(cluster_number):
         if value==0:
            cluster_number[index]=1
    print(cluster_number)
    num_cluster = sum(cluster_number)
    print(num_cluster)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.MoCo(
        models.__dict__[args.arch], num_cluster,
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    print(model)
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    
   
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.weight_decay)

    
    # optionally resume from a checkpoint
    if args.resume:
        print(os.path.isfile(args.resume))
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    cudnn.benchmark = True 
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    train_loader_cluster = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size*5, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    pretrain_epochs = args.epochs // 2
    tsc_epochs = args.epochs - pretrain_epochs
 
    for epoch in range(args.start_epoch, pretrain_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, pretrain_epochs, args)
        criterion = KCL(K=args.moco_k,k=6,temperature=args.moco_t).cuda()
        train(train_loader, model, criterion, optimizer,epoch,args)
        
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0) :
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename='Imagenet/last.pth.tar')
                if (epoch + 1) % 100== 0:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, is_best=False, filename='Imagenet/checkpoint_{:04d}.pth.tar'.format(epoch))
         
           
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch,tsc_epochs):
        if args.distributed:
                train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, tsc_epochs, args)
        criterion = SupConLoss_rank(K=args.moco_k,temperature=args.moco_t).cuda() 
        if epoch % args.step == 0:
            targets=cluster(train_loader_cluster,model,cluster_number,args)
            train_dataset.new_labels = targets  
        train(train_loader, model, criterion, optimizer,epoch+pretrain_epochs,args)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename='Imagenet/last.pth.tar')
                if (epoch + 1) % 100== 0:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, is_best=False, filename='Imagenet/cclcheckpoint_{:04d}.pth.tar'.format(epoch+pretrain_epochs))
def cluster (train_loader_cluster,model,cluster_number,args):
    model.eval()
    features_sum = []
    print('cluster_proccess')
    for i, (images, target, index) in enumerate(train_loader_cluster):
        images = images[0].cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        if i % 100 ==0:
              print(target)
        with torch.no_grad():
            features = model(im_q=images)
            features = features.detach()
            features_sum.append(features)
    features= torch.cat(features_sum,dim=0)
    features = torch.split(features,args.cls_num_list, dim=0)
    target = [[] for i in range(len(cluster_number))]
    for i in range(len(cluster_number)):  
        if cluster_number[i] >1:
            cluster_ids_x, cluster_centers  = kmeans(X=features[i], num_clusters=cluster_number[i], tol=1e-3, distance='cosine', device=torch.device("cuda"))
            target[i]=cluster_ids_x
        else:
            target[i] = torch.zeros(1,features[i].size()[0], dtype=torch.int).squeeze(0)
        if i% 100 ==0:
            print(target[i])
    cluster_number_sum=[sum(cluster_number[:i]) for i in range(len(cluster_number))]
    for i ,k in enumerate(cluster_number_sum):
         target[i] =  torch.add(target[i], k)
    targets=torch.cat(target,dim=0)
    targets = targets.numpy().tolist()
    return targets


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images,target,cluster_target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            cluster_target = cluster_target.cuda(args.gpu, non_blocking=True)
        
        
        # compute output
        logits,labels,true_labels = model(im_q=images[0], im_k=images[1], labels=cluster_target,true_labels=target)
        if epoch < args.epochs//2:
            loss = criterion(logits,target,true_labels)
        else:
            loss = criterion(logits,target,true_labels,cluster_target,labels)                    
        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, total_epochs, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / total_epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



if __name__ == '__main__':
    main()
