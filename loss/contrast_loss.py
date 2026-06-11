"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
import torch
import torch.nn as nn


class SampleConLoss(nn.Module):
    def __init__(self, temperature=0.07, sample_stride=2, contrastive_method='gcl'):
        super(SampleConLoss, self).__init__()
        self.temp = temperature
        self.supconloss = SupConLoss(temperature=self.temp, contrastive_method=contrastive_method)
        self.stride = sample_stride

    def forward(self, features, labels):
        features = features[:: self.stride, :, :]
        labels = labels[:: self.stride]
        return self.supconloss(features, labels)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR."""

    def __init__(
        self,
        threshold=0.1,
        temperature=0.07,
        contrast_mode='all',
        base_temperature=0.07,
        contrastive_method='simclr',
    ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.threshold = threshold
        self.contrastive_method = contrastive_method

    def _cosine_simililarity(self, x, y):
        return self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))

    def forward(self, features, labels=None, mask=None):
        device = features.device
        original_mask = mask

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
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
            if self.contrastive_method in ['gcl']:
                mask = torch.eq(labels, labels.T).float().to(device)
            elif self.contrastive_method in ['pcl', 'wpcl', 'wcl', 'superpixel_pcl']:
                mask = (
                    torch.abs(labels.T.repeat(batch_size, 1) - labels.repeat(1, batch_size)) < self.threshold
                ).float().to(device)
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

        logits = torch.div(
            self._cosine_simililarity(anchor_feature, contrast_feature),
            self.temperature,
        )

        if labels is None and original_mask is not None:
            mask = mask
        else:
            mask = mask.repeat(anchor_count, contrast_count)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
