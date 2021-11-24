import argparse
import os

def str2bool(x):
    return x.lower() in ('true')

parser = argparse.ArgumentParser('AdaPC Flags')

# training
parser.add_argument('--batch_size', type=int, default=24, help='batch size for training')
parser.add_argument('--epoch',  default=300, type=int, help='number of epoch for training')
parser.add_argument('--learning_rate', default=0.001, type=float, help='classifier learning rate')
parser.add_argument('--learning_rate_a', default=0.001, type=float, help='augmentator learning rate for training')
parser.add_argument('--no_decay', type=str2bool, default=False) # forgot
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
parser.add_argument('--pretrain', type=str, default=None, help='whether use pretrain Augment')
parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of parameters (weight decay)')
parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate of learning rate')

parser.add_argument('--restore', action='store_true')
parser.add_argument('--gpu', type=str, default='0', help='specify which gpu to use')
parser.add_argument('--num_class', type=int, default=40)
parser.add_argument('--num_points', type=int, default=1024)
parser.add_argument('--use_normal', type=str2bool, default=False)
parser.add_argument('--log_dir', default='log', help='log_dir')
parser.add_argument('--data_dir', default='Dataset folder')
parser.add_argument('--epoch_per_save', type=int, default=5)


# augmentation types and regularization flags
parser.add_argument('--apply_scale', type=str2bool, default=True, help='apply_scale in augmenter')
parser.add_argument('--apply_shift', type=str2bool, default=True, help='apply shift in augmenter')
parser.add_argument('--apply_rot', type=str2bool, default=True, help='apply rotation in augmenter')
parser.add_argument('--apply_noise', type=str2bool, default=False, help='apply noise in augmentor')
parser.add_argument('--aug_dropout', type=str2bool, default=True, help='apply dropout for augmentations with threshold of 0.5')
parser.add_argument('--apply_reg', type=str2bool, default=True, help='whether to apply regularization to constrain augmenter output')
parser.add_argument('--reg_weight', type=float, default=0.5, help='weight of augloss regularization')
parser.add_argument('--reg_weight_J', type=float, default=1.0, help='weight of augloss regularization for jittering')

opts = parser.parse_args()
