import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)   
    
class KCL(nn.Module):
    def __init__(self,K,k=6,temperature=0.07):
        super(KCL, self).__init__()
        self.K =K
        self.k = k
        self.temperature = temperature
    def forward(self, logits,im_labels,queue_label):
        logits = logits /self.temperature
        im_labels = im_labels.contiguous().view(-1, 1)
        mask = torch.eq(im_labels, queue_label).float()
        mask_pos_view = torch.zeros_like(mask)
        # positive logits from queue
        num_positive=self.k
        if num_positive > 0:
            for i in range(num_positive):
                all_pos_idxs = mask.view(-1).nonzero().view(-1)
                num_pos_per_anchor = mask.sum(1)
                num_pos_cum = num_pos_per_anchor.cumsum(0).roll(1)
                num_pos_cum[0] = 0
                rand = torch.rand(mask.size(0), device=mask.device)
                idxs = ((rand * num_pos_per_anchor).floor() + num_pos_cum).long()
                idxs = idxs[num_pos_per_anchor.nonzero().view(-1)]
                sampled_pos_idxs = all_pos_idxs[idxs.view(-1)]
                mask_pos_view.view(-1)[sampled_pos_idxs] = 1
        else:
            mask_pos_view = mask.clone()
        mask_pos_view_class = mask_pos_view.clone()
        #print(queue_label.size(1))
        mask_pos_view_class[:, queue_label.size(1):] = 0
        
        mask_pos_view = torch.cat([torch.ones([mask_pos_view.shape[0], 1]).cuda(), mask_pos_view], dim=1)
        mask_pos_view_class = torch.cat([torch.ones([mask_pos_view_class.shape[0], 1]).cuda(), mask_pos_view_class], dim=1)
        log_prob = F.normalize(logits.exp(), dim=1, p=1).log()
        loss = - torch.sum((mask_pos_view_class * log_prob).sum(1) / mask_pos_view.sum(1)) / mask_pos_view.shape[0]
        return loss
    
class SupConLoss_ccl(nn.Module):
    def __init__(self,K,gramma=0.2,temperature=0.07):
        super(SupConLoss_ccl, self).__init__()
        self.K = K
        self.gramma = gramma
        self.temperature = temperature
    def forward(self, logits, im_label,queue_label,im_cluster,queue_cluster):

        logits =logits /self.temperature
        log_prob = F.normalize(logits.exp(), dim=1, p=1).log()

        im_label = im_label.contiguous().view(-1, 1)
        label_mask = torch.eq(im_label, queue_label).float()
        label_mask = label_mask.clone()
        im_cluster = im_cluster.contiguous().view(-1, 1)
        cluster_mask = torch.eq(im_cluster, queue_cluster).float()
        cluster_mask = cluster_mask.clone()
        # compute logits
        label_mask = torch.cat([torch.ones([label_mask.shape[0], 1]).cuda(), label_mask], dim=1)
        cluster_mask = torch.cat([torch.ones([cluster_mask.shape[0], 1]).cuda(), cluster_mask], dim=1)
        loss_cluster = - torch.sum((cluster_mask * log_prob).sum(1) / cluster_mask.sum(1)) / cluster_mask.shape[0]
        loss_label= - torch.sum((label_mask * log_prob).sum(1) / label_mask.sum(1)) / label_mask.shape[0]
        # loss      
        loss =loss_cluster + self.gramma*loss_label
        return loss

class SupConLoss_rank(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self,K,temperature=0.07,ranking_temperature=0.12,grama=0.2):
        super(SupConLoss_rank, self).__init__()
        self.temperature = temperature
        self.ranking_temperature = ranking_temperature
        self.grama = grama
        self.K = K
    def forward(self, logits,im_labels,queue_label,im_cluster,queue_cluster):

        im_labels = im_labels.contiguous().view(-1, 1)
        label_mask = torch.eq(im_labels, queue_labels).float()
        
        im_cluster = im_cluster.contiguous().view(-1, 1)
        cluster_mask = torch.eq(im_cluster, queue_cluster).float()
        cluster_mask = torch.cat([torch.ones([cluster_mask.shape[0], 1]).cuda(), cluster_mask], dim=1)
        logits_cluster = logits/self.temperature
        log_cluster_prob = F.normalize(logits_cluster.exp(), dim=1, p=1).log()
        loss_cluster = - torch.sum((cluster_mask * log_cluster_prob).sum(1) / cluster_mask.sum(1)) / cluster_mask.shape[0]

        Bool = ~cluster_mask.bool()
        Inverse_cluster =Bool.float()
        label_mask = Inverse_cluster * label_mask
        label_mask = torch.cat([torch.ones([label.shape[0], 1]).cuda(), label_mask], dim=1)
        Inverse_cluster = torch.cat([torch.ones([Inverse_cluster.shape[0], 1]).cuda(), Inverse_cluster], dim=1)
        logits_label = logits/self.ranking_temperature
        log_inverse_cluster = torch.exp(logits_label) * Inverse_cluster
        log_label_prob = logits_label - torch.log(log_inverse_cluster .sum(1, keepdim=True))
        loss_label = - torch.sum((label_mask* log_label_prob).sum(1) / label_mask.sum(1)) / label_mask.shape[0]
        loss =loss_cluster + self.gramma*loss_label

        return loss   

class SupConLoss_warm(nn.Module):
    def __init__(self,K,gramma=0.2, temperature=0.07, 
                 base_temperature=None):
        super(SupConLoss_warm, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.gramma = gramma
    def forward(self, features, labels, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        ss = features.shape[0]
        batch_size = (features.shape[0] - self.K ) // 2

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask 

        exp_log_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_log_logits .sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # loss
      
        loss =-(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
