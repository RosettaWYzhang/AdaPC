
import os
import glob
import random
import logging
from datetime import datetime
import pprint
pp = pprint.PrettyPrinter()

import numpy as np
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import meta_utils
import loss_utils
from config import opts
from pointnet import PointNetCls
from augmentor import Augmentor
from PointCloudDataLoader import ModelNetDataLoader

# Set random seed for reproducibility
seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Model:
    def __init__(self, opts):
        self.opts = opts
        self.set_logger()

    def set_logger(self):
        self.logger = logging.getLogger("CLS")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.opts.log_dir, "log_train.txt"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def train(self):

        self.log_string('PARAMETER ...')
        self.log_string(self.opts)
        with open(os.path.join(self.opts.log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(self.opts)):
                log.write(arg + ': ' + str(getattr(self.opts, arg)) + '\n')  # log of arguments
        writer = SummaryWriter(logdir=self.opts.log_dir)

        '''DATA LOADING'''
        self.log_string('Load dataset ...')
        trainval_dataset = ModelNetDataLoader(self.opts, partition='train')
        train_length = int(len(trainval_dataset) * 0.9)
        val_length = len(trainval_dataset) - train_length
        train_set, val_set = torch.utils.data.random_split(trainval_dataset, [train_length, val_length])
        trainDataLoader = DataLoader(train_set, batch_size=self.opts.batch_size, shuffle=True, drop_last=False)
        valDataLoader = DataLoader(val_set, batch_size=self.opts.batch_size, shuffle=True, drop_last=False)
        val_iter = iter(valDataLoader)
        train_iter = iter(trainDataLoader)
        testDataLoader = DataLoader(ModelNetDataLoader(self.opts, partition='test'),
                                 batch_size=self.opts.batch_size, shuffle=False,)
        self.log_string("The number of training data is: %d" % len(trainDataLoader.dataset))
        self.log_string("The number of test data is: %d" % len(testDataLoader.dataset))

        '''MODEL LOADING'''
        num_class = self.opts.num_class
        self.dim = 3 if self.opts.use_normal else 0
        mse_fn = nn.MSELoss()
        classifier = PointNetCls(num_class).cuda()
        augmentor = Augmentor(self.opts.apply_scale, self.opts.apply_shift, self.opts.apply_rot, self.opts.apply_noise, self.opts.aug_dropout).cuda()

        if self.opts.restore:
            self.log_string('Use pretrain Augment...')
            checkpt = sorted(glob.glob(os.path.join(self.opts.log_dir,'pointnet-*.pth')))
            checkpoint = torch.load(checkpt[-1]) # restore from the last epoch, instead of the best one
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            augmentor.load_state_dict(checkpoint['augmentor_state_dict'])
        else:
            print('No existing Augment, starting training from scratch...')
            start_epoch = 0

        optimizer_c = torch.optim.Adam(
            classifier.parameters(),
            lr=self.opts.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.opts.decay_rate
        )

        optimizer_a = torch.optim.Adam(
            augmentor.parameters(),
            lr=self.opts.learning_rate_a,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.opts.decay_rate
        )

        scheduler_c = torch.optim.lr_scheduler.StepLR(optimizer_c, step_size=20, gamma=self.opts.lr_decay)
        scheduler_a = None
        global_epoch = 0
        best_tst_accuracy = 0.0
        best_epoch = 0
        ispn = True
        self.logger.info('Apply loss smoothing?...' + str(ispn))

        '''TRANING'''
        self.logger.info('Start training...')
        for epoch in range(start_epoch, self.opts.epoch):
            self.log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, self.opts.epoch))
            if scheduler_c is not None:
                scheduler_c.step(epoch)
            if scheduler_a is not None:
                scheduler_a.step(epoch)
            batch_id = 0
            train_loss = 0.0
            while True:
                # get one batch of validation data
                try:
                    points_val, target_val = next(val_iter)
                except StopIteration:
                    val_iter = iter(valDataLoader)
                    points_val, target_val = next(val_iter)
                target_val = target_val[:, 0]
                points_val, target_val = points_val.cuda(), target_val.cuda().long()
                points_val = points_val.transpose(2, 1).contiguous()
                # get one batch of training data
                try:
                    points, target = next(train_iter)
                    batch_id += 1
                except StopIteration:
                    # one epoch ends, reiniterlize train val split, break
                    train_set, val_set = torch.utils.data.random_split(trainval_dataset, [train_length, val_length])
                    trainDataLoader = DataLoader(train_set, batch_size=self.opts.batch_size, shuffle=True, drop_last=False)
                    valDataLoader = DataLoader(val_set, batch_size=self.opts.batch_size, shuffle=True, drop_last=False)
                    val_iter = iter(valDataLoader)
                    train_iter = iter(trainDataLoader)
                    break  # break when we finish one epoch

                target = target[:, 0]
                points, target = points.cuda(), target.cuda().long()
                points = points.transpose(2, 1).contiguous()

                classifier = classifier.train()
                augmentor = augmentor.train()

                # *** BILEVEL OPTIMIZATION TO TUNE AUGMENTOR ***
                optimizer_a.zero_grad()
                optimizer_c.zero_grad()
                pred_pc, pc_tran, pc_feat = classifier(points)
                aug_pc, trans = augmentor(points)
                pred_aug, aug_tran, aug_feat = classifier(aug_pc)
                loss_unroll = loss_utils.cls_loss(pred_pc, pred_aug, target, pc_tran, aug_tran, pc_feat, aug_feat, ispn=ispn)
                # one step gradient descent on training data
                unrolled_model = meta_utils.compute_unrolled_model(classifier, loss_unroll, optimizer_c, self.opts)
                pred_val, val_tran, val_feat = unrolled_model(points_val)
                clsLoss_val = loss_utils.cls_loss_simple(pred_val, target_val, val_tran, ispn)
                optimizer_a.zero_grad()
                clsLoss_val.backward(retain_graph=True)

                cls_grads = [v.grad.data.detach() for v in unrolled_model.parameters()]
                implicit_grads = meta_utils.finite_diff(augmentor,
                                                        classifier,
                                                        points,
                                                        aug_pc,
                                                        target,
                                                        cls_grads,
                                                        ispn=ispn,
                                                        ) # grad val w.r.t. hyperparameters

                # add regularization
                aug_reg = 0
                if self.opts.apply_reg and bool(trans):
                    zero_reg = torch.zeros(points.shape[0], 1).cuda()
                    one_reg = torch.ones(points.shape[0], 1).cuda()
                    jitter_reg = zero_reg
                    for name, value in trans.items():
                        if 'scale' in name:
                            aug_reg += self.opts.reg_weight * mse_fn(value, one_reg)
                        elif 'noise_range' in name:
                            aug_reg += self.opts.reg_weight_J * mse_fn(value, jitter_reg)
                        else: # shifting and rot_y
                            aug_reg += self.opts.reg_weight * mse_fn(value, zero_reg)
                    if self.opts.apply_reg and aug_reg != 0:
                        dalpha = list(torch.autograd.grad(aug_reg, augmentor.parameters(), allow_unused=True))
                        for count, ig in enumerate(implicit_grads):
                            if ig is not None and dalpha[count] is not None:
                                dalpha[count] -= ig
                            elif ig is not None and dalpha[count] is None:
                                # this branch is possible as reg is only dependent on mu, not sigma
                                dalpha[count] = -ig
                    else:
                        dalpha = []
                        for ig in implicit_grads:
                            if ig is None:
                                dalpha += [None]
                            else:
                                dalpha += [-ig]

                    for n, (v, g) in enumerate(zip(augmentor.parameters(), dalpha)):
                        if v.grad is None:
                            if not (g is None):
                                v.grad = Variable(g.data)
                        else: # v.grad is not None
                            if not (g is None):
                                v.grad.data.copy_(g.data)
                    optimizer_a.step()

                # update classifier
                pred_pc, pc_tran, pc_feat = classifier(points) # get original prediction
                optimizer_c.zero_grad()
                aug_pc, trans = augmentor(points)
                pred_aug, aug_tran, aug_feat = classifier(aug_pc)
                clsLoss = loss_utils.cls_loss(pred_pc, pred_aug, target, pc_tran, aug_tran, pc_feat,
                                              aug_feat, ispn=ispn)
                clsLoss.backward()
                optimizer_c.step()
                train_loss += clsLoss.detach().cpu().numpy()

            if batch_id % 100 == 0:
                print('current epoch is %d and current batch id is %d' %(epoch, batch_id))
                print('current loss is %f' %clsLoss.detach().cpu().numpy())

            with torch.no_grad():
                train_acc_noaug, train_loss_noaug = self.eval_one_epoch(classifier.eval(), trainDataLoader, ispn=ispn)
                test_acc, test_loss = self.eval_one_epoch(classifier.eval(), testDataLoader, ispn=ispn)

            self.log_string('Train Loss with augmentation: %.2f'%(train_loss/batch_id))
            self.log_string('Train Loss without augmentation: %.2f'%(train_loss_noaug))
            self.log_string('Train Accuracy without augmentation: %.2f'%(train_acc_noaug))

            self.log_string('Test loss: %f' %test_loss)
            self.log_string('Test Accuracy: %f' %test_acc)
            self.log_string('Best test acc: %f achieved at epoch %d' %(best_tst_accuracy, best_epoch))
            writer.add_scalar("Test_Acc", test_acc, epoch)

            save_epoch = test_acc > best_tst_accuracy and test_acc >= 0.90
            if test_acc > best_tst_accuracy:
                best_epoch = epoch
                best_tst_accuracy = test_acc
            if save_epoch or (epoch == self.opts.epoch-1):
                self.log_string('Save model...')
                self.save_checkpoint(
                    global_epoch + 1,
                    test_acc,
                    classifier,
                    augmentor,
                    optimizer_c,
                    optimizer_a,
                    str(self.opts.log_dir))

            global_epoch += 1
        self.log_string('Best Test Accuracy: %f' % best_tst_accuracy)
        self.log_string('End of training...')
        self.log_string(self.opts.log_dir)
        print('reached the end of training 1 epoch')


    def eval_one_epoch(self, model, loader, ispn):
        mean_correct = []
        test_pred = []
        test_true = []
        test_loss = 0
        for j, data in enumerate(loader, 0):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = model.eval()
            pred, tran, feat = classifier(points)
            test_loss += loss_utils.cls_loss_simple(pred, target.long(), tran, ispn=ispn)
            pred_choice = pred.data.max(1)[1]
            test_true.append(target.cpu().numpy())
            test_pred.append(pred_choice.detach().cpu().numpy())

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        test_loss = test_loss.detach().cpu().numpy() / j
        return test_acc, test_loss

    def save_checkpoint(self, epoch, test_accuracy, model, augmentor, optimizer, optimizer_a, path, prefix='checkpoint'):
        savepath = path + '/%s-%f-%04d.pth' % (prefix, test_accuracy, epoch)
        print(savepath)
        state = {
            'epoch': epoch,
            'test_accuracy': test_accuracy,
            'augmentor_state_dict': augmentor.state_dict() if augmentor is not None else None,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_a_state_dict': optimizer_a.state_dict() if optimizer_a is not None else None,
        }
        torch.save(state, savepath)

    def log_string(self, msg):
        print(msg)
        self.logger.info(msg)

if __name__ == '__main__':
    opts.log_dir = os.path.join('log', opts.log_dir)
    if not os.path.exists(opts.log_dir):
        os.makedirs(opts.log_dir)
    print('checkpoints:', opts.log_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu
    model = Model(opts)
    model.train()
