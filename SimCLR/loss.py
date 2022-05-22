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
    
    
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self,temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
       
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature/ self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

class SupConLoss_ccl(nn.Module):
    def __init__(self,temperature=0.1, grama=0.25, 
                 base_temperature=0.07):
        super(SupConLoss_ccl, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.grama = grama
    def forward(self,features,labels,cluster_target, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        cluster_target = cluster_target.contiguous().view(-1, 1)
        cluster_mask = torch.eq(cluster_target, cluster_target.T).float().to(device)
        cluster_mask = cluster_mask.repeat(anchor_count, contrast_count)
        cluster_mask = cluster_mask * logits_mask
      
        # for numerical stability
        mean_log_label_pos = (cluster_mask * log_prob).sum(1) / cluster_mask.sum(1)
        # loss
       
        loss = - (self.temperature / self.base_temperature) * mean_log_label_pos-self.grama*(mean_log_prob_pos)
        loss = loss.mean()
        return loss
    


class SupConLoss_rank(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self,num_class, temperature=0.1,ranking_temperature=0.2, contrast_mode='all',grama=0.25,
                 base_temperature=0.07):
        super(SupConLoss_rank, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.ranking_temperature = ranking_temperature
        self.num_class = num_class
        self.grama = grama
    def forward(self,features,labels,cluster_target, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        cluster_target = cluster_target.contiguous().view(-1, 1)
        if cluster_target.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(cluster_target, cluster_target.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        one_hot_labels=torch.nn.functional.one_hot(labels, num_classes=self.num_class)
        one_hot_labels = one_hot_labels.repeat(anchor_count,1).float().cuda()
        ranking_temperature = torch.from_numpy(self.ranking_temperature).float().cuda()
        ranking_temperature = torch.matmul(one_hot_labels, ranking_temperature.T)
        ranking_temperature = ranking_temperature.unsqueeze(0).T
        anchor_rank_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            ranking_temperature)
        # for numerical stability
        logits_rank_max, _ = torch.max(anchor_rank_contrast, dim=1, keepdim=True)
        logits_rank = anchor_rank_contrast - logits_rank_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        # compute mean of log-likelihood over positive
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        labels = labels.contiguous().view(-1, 1)
        
        label_mask = torch.eq(labels, labels.T).float().to(device)
        label_mask = label_mask.repeat(anchor_count, contrast_count)
        Bool = ~mask.bool()
        Inverse_cluster_target =Bool.float()
        logits_label_mask = Inverse_cluster_target * logits_mask  
        list=np.arange(batch_size,batch_size* anchor_count).tolist()
        list.extend(np.arange(batch_size).tolist())
        ad_mask = torch.scatter(
            torch.zeros(mask.size(0),mask.size(1)),
            0,
            torch.tensor([list]),
            1
        )
        ad_logits_label_mask= ad_mask.cuda()+logits_label_mask
        exp_logits_rank = torch.exp(logits_rank) * ad_logits_label_mask
        
        label_mask=logits_label_mask*label_mask
        label_mask=ad_mask.cuda()+label_mask
        # compute mean of log-likelihood over positive
        log_prob_rank = logits_rank - torch.log(exp_logits_rank.sum(1, keepdim=True))

        mean_log_label_pos = (label_mask * log_prob_rank).sum(1) /label_mask.sum(1) 
        '''
        label_mask = torch.eq(labels, labels.T).float().to(device)
        label_mask = label_mask.repeat(anchor_count, contrast_count)
        label_mask = label_mask * logits_mask
        exp_logits_rank = torch.exp(logits_rank) * logits_mask
        log_prob_rank = logits_rank - torch.log(exp_logits_rank.sum(1, keepdim=True))
        
        mean_log_label_pos = (label_mask * log_prob_rank).sum(1) /label_mask.sum(1) 
        '''
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos-self.grama*(mean_log_label_pos)
        loss = loss.mean()

        return loss
 
