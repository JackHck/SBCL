# Subclass-balancing-contrastive-learning
This repository provides the  code for paper: <br>
**Subclass-balancing Contrastive Learning for Long-tailed Recognition**

## Overview
In this paper, we prospose subclass-balancing contrastive learning (SBCL),
a novel supervised contrastive learning defined on subclasses, which are the clusters within each
head class, have comparable size as tail classes, and are adaptively updated during the training.
Instead of sacrificing instance-balance for class-balance, our method achieves both instance- and
subclass-balance by exploring the head-class structure in the learned representation space of the
model-in-training. In particular, we propose a bi-granularity contrastive loss that enforces a sample
(1) to be closer to samples from the same subclass than all the other samples; and (2) to be closer to
samples from a different subclass but the same class than samples from any other subclasses. While
the former learns representations with balanced and compact subclasses, the latter preserves the class
structure on subclass level by encouraging the same class’s subslasses to be closer to each other than
to any different class’s subclasses. Hence, it can learn an accurate classifier distinguishing original
classes while enjoy both the instance- and subclass-balance.
## Requiremenmts
* ImageNet dataset
* Python ≥ 3.6
* PyTorch ≥ 1.4
* pip install kmeans_pytorch (the iteration be set at less than 25)
## CIFAR dataset
The code will help you download the CIFAR dataset.
### First-stage train
To perform SBCL using 2-gpu machines, run:
<pre>python main_pcl.py \ 
  -a resnet50 \ 
  --lr 0.03 \
  --batch-size 256 \
  --temperature 0.2 \
  --mlp --aug-plus --cos (only activated for PCL v2) \	
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --exp-dir experiment_pcl
  [Imagenet dataset folder]
</pre>
### Second-stage train
To evalute the representation learning, run
<pre>python eval_cls_imagenet.py --pretrained [your pretrained model] \
  -a resnet50 \ 
  --lr 5 \
  --batch-size 256 \
  --id ImageNet_linear \ 
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [Imagenet dataset folder]
</pre>


