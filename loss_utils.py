#!/usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:liruihui
@file: loss_utils.py
@time: 2019/09/23
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description:
"""
import numpy as np
import torch
import torch.nn.functional as F
import random


def cal_loss_raw(pred, gold):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)
    eps = 0.2
    n_class = pred.size(1)
    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)
    loss_raw = -(one_hot * log_prb).sum(dim=1)
    loss = loss_raw.mean()
    return loss,loss_raw


def mat_loss(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


def cls_loss(pred, pred_aug, gold, pc_tran, aug_tran, pc_feat, aug_feat, ispn):
    ''' Calculate cross entropy loss, apply label smoothing if needed.
        # three components to the loss function:
        # 1. cross entropy;
        # 2. feature transformation regularization;
        # 3. feature difference between original and augmented
    '''

    mse_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    cls_pc, cls_pc_raw = cal_loss_raw(pred, gold)
    cls_aug, cls_aug_raw = cal_loss_raw(pred_aug, gold)

    if ispn:
        cls_pc = cls_pc + 0.001*mat_loss(pc_tran)
        cls_aug = cls_aug + 0.001*mat_loss(aug_tran)

    if pc_feat is not None:
        feat_diff = 10.0*mse_fn(pc_feat,aug_feat)
    else:
        feat_diff = 0

    cls_loss = cls_pc + cls_aug  + feat_diff
    return cls_loss


def cls_loss_simple(pred, gold, pc_tran, ispn=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    cls_pc, _ = cal_loss_raw(pred, gold)
    if ispn:
        cls_pc = cls_pc + 0.001*mat_loss(pc_tran)
    cls_loss = cls_pc
    return cls_loss
