import numpy as np
import torch
import loss_utils
from pointnet import PointNetCls

def finite_diff(augmentor, classifier, pc, aug_pc, target, cls_grads, ispn=True, r=1e-2,):
    '''
    Args:
        augmentor: augmentation model model
        classifier: classifier model
        pc: original point clouds
        aug_pc: augmented point clouds
        target: augmented label if mixed
        cls_grads: grad of val w.r.t. classifier model parameters
    Returns: implicit_grads of augmentor wrt validation loss
    '''
    R = r / (concat(cls_grads).data.detach().norm())
    # R is the epsilon in equation
    # perturb classifier parameters
    for p, v in zip(classifier.parameters(), cls_grads):
       p.data.add_(R, v)   # p is w in equations
    # run -classifier and compute loss
    pred1_ori, trans1_ori, feat1_ori = classifier(pc)
    pred1_aug, trans1_aug, feat1_aug = classifier(aug_pc)
    loss_add = loss_utils.cls_loss(pred1_ori, pred1_aug, target, trans1_ori, trans1_aug, feat1_ori, feat1_aug, ispn=ispn)
    grads_p = torch.autograd.grad(loss_add, augmentor.parameters(), retain_graph=True, allow_unused=True)
    for p, v in zip(classifier.parameters(), cls_grads):
        p.data.sub_(2 * R, v)
    pred2_ori, trans2_ori, feat2_ori = classifier(pc)
    pred2_aug, trans2_aug, feat2_aug = classifier(aug_pc)
    loss_sub = loss_utils.cls_loss(pred2_ori, pred2_aug, target, trans2_ori, trans2_aug, feat2_ori, feat2_aug, ispn=ispn)
    grads_n = torch.autograd.grad(loss_sub, augmentor.parameters(), retain_graph=True, allow_unused=True)
    # restore classification parameters
    for p, v in zip(classifier.parameters(), cls_grads):
        p.data.add_(R, v)
    return [ None if ( x is None ) or ( y is None) else (x - y).div_(2 * R) for x, y in zip(grads_p, grads_n) ]


def compute_unrolled_model(model, loss, network_optimizer, opts):
    eta = opts.learning_rate
    theta = concat(model.parameters()).data
    dtheta = concat(torch.autograd.grad(loss, model.parameters())).data + opts.decay_rate * theta
    try:
        beta1 = 0.9
        beta2 = 0.999
        steps = [network_optimizer.state[v]['step'] for v in model.parameters()][0]
        exp_avg = concat(network_optimizer.state[v]['exp_avg'] for v in model.parameters()).mul_(beta1).add_(dtheta.mul_(1-beta1))
        exp_avg.div_(1-beta1 ** steps)
        exp_avg_sq = concat(network_optimizer.state[v]['exp_avg_sq'] for v in model.parameters()).mul_(beta2).add_((dtheta * dtheta).mul_(1-beta2))
        exp_avg_sq.div_(1-beta2 ** steps)

    except:
        print('no momentum buffer or exp_avg in optimizer; initialize as zero')
        moment = torch.zeros_like(theta)
        exp_avg = torch.zeros_like(theta)
        exp_avg_sq = torch.zeros_like(theta)

    theta.sub(eta, exp_avg / (torch.sqrt(exp_avg_sq) + 1e-8))
    unrolled_model = construct_model_from_theta(model, opts, theta, opts.num_class)
    return unrolled_model


def construct_model_from_theta(model, opts, theta, num_class):
    # model is the classifier
    model_new = PointNetCls(num_class).cuda()
    model_dict = model.state_dict()

    params, offset = {}, 0
    for k, v in model.named_parameters():
        v_length = np.prod(v.size())
        params[k] = theta[offset: offset + v_length].view(v.size())
        offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()


def concat(xs):
    return torch.cat([x.view(-1) for x in xs])
