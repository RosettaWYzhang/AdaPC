import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.nn.functional as F
import numpy as np
import random


class Augmentor(nn.Module):
    '''generate transformation vector from noise input

    Args:
        apply_scale: boolean variable, whether to apply scale transformation
        apply_shift: boolean variable, whether to apply shift transformation
        apply_rot: boolean variable, whether to apply rotation transformation
        apply_noise: boolean variable, whether to apply jittering transformation
        aug_dropout: boolean variable, whether to apply augmentation dropout to create more variations
    Returns:
        A 1* dim transformation vector (1 * 3 for rotation, 1 * 3 for translation and 1 * 1 for y-axis rotation
    '''
    def __init__(self, apply_scale, apply_shift, apply_rot, apply_noise, aug_dropout):
        super(Augmentor, self).__init__()
        self.dim = 0
        self.apply_scale = apply_scale
        self.apply_shift = apply_shift
        self.apply_rot = apply_rot
        self.apply_noise = apply_noise
        self.aug_dropout = aug_dropout
        # order: rotate, scale, shift
        if self.apply_rot:
            # rotation is only applied at the y axis
            self.rot_min_index = self.dim
            self.dim += 1
        if self.apply_scale:
            self.scale_min_index = self.dim
            self.dim += 3
        if self.apply_shift:
            self.shift_min_index = self.dim
            self.dim += 3
        if self.apply_noise:
            self.noise_min_index = self.dim
            self.dim += 1
        self.fc1 = nn.Linear(9, 9)
        self.fc2 = nn.Linear(9, self.dim)
        self.bn1 = nn.BatchNorm1d(9)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, points):
        '''
        Args:
            points: a cuda tensor of shape [B, 3, N]
        return:
            aug_pc: a cuda tensor of shape [B, 3, N],
            aug_stats: a dictionary of transformation applied
        '''
        aug_stats = {}
        batchsize = points.size()[0]
        points = points.transpose(2, 1)
        noise = torch.randn(batchsize, 9).cuda()
        x = F.relu(self.bn1(self.fc1(noise)))
        x = self.fc2(x)
        if self.apply_rot:
            rotation = x[:, self.rot_min_index][:, None, None]
        if self.apply_scale:
            scale = x[:, self.scale_min_index:self.scale_min_index+3][:, None, :]
        if self.apply_shift:
            shift = x[:, self.shift_min_index:self.shift_min_index+3][:, None, :]
        if self.apply_noise:
            noise_range = x[:, self.noise_min_index][:, None, None]
        tensor_0 = torch.zeros(batchsize).cuda()
        tensor_1 = torch.ones(batchsize).cuda()
        p1 = random.random()
        p2 = random.random()
        p3 = random.random()
        p4 = random.random()
        if self.aug_dropout:
            thres = 0.5
        else:
            thres = 0.0

        if self.apply_rot:
            angleY = rotation.squeeze(-1).squeeze(-1)
            aug_stats['rot_y'] = angleY[:, None]
            R = torch.stack([
                torch.stack([torch.cos(angleY), tensor_0, torch.sin(angleY)], -1),
                torch.stack([tensor_0, tensor_1, tensor_0], -1),
                torch.stack([-torch.sin(angleY), tensor_0, torch.cos(angleY)], -1)], 1).reshape(batchsize, 3,3)
            if p1 > thres:
                points = torch.bmm(points, R)
        if self.apply_scale:
            aug_stats['scale_x'] = scale[:, :, 0]
            aug_stats['scale_y'] = scale[:, :, 1]
            aug_stats['scale_z'] = scale[:, :, 2]
            if p2 > thres:
                points = points * scale

        if self.apply_shift:
            aug_stats['shift_x'] = shift[:, :, 0]
            aug_stats['shift_y'] = shift[:, :, 1]
            aug_stats['shift_z'] = shift[:, :, 2]
            if p3 > thres:
                points = points + shift
        if self.apply_noise:
            noise_squeeze = noise_range.squeeze(-1)
            aug_stats['noise_range'] = noise_squeeze
            if p4 > thres:
                point_shift = (torch.rand(batchsize, 1024).cuda() * 0.002 + noise_squeeze)[:, :, None]
                points = points + point_shift
        return points.transpose(2, 1), aug_stats
