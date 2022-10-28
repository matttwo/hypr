# -*- coding: utf-8 -*-
import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm

import copy

from dataset.inaturalist import INat_Birds
from models.resnet import ResNet, resnet18

parser = argparse.ArgumentParser(description='Trains an iNaturalist classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, choices=['iNat'], default='iNat',
                    help='Choose between iNat.')
parser.add_argument('--model', '-m', type=str, default='resnet18',
                    choices=['resnet18', 'hnn'], help='Choose architecture.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')

# HNN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')

# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/hyperbolic', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')

# Hyperbolic related arguments
#######################################################################
parser.add_argument("--use-hyperbolic", action="store_true", default=False)

parser.add_argument(
    "--c", type=float, default=1.0, help="Curvature of the Poincare ball"
)
parser.add_argument(
    "--dim", type=int, default=128, help="Dimension of the Poincare ball"
)

parser.add_argument(
    "--max_clip_norm", type=float, default=15.0, help="Max clip norm of the Euclidean embedding"
)

parser.add_argument('--clip_norm_type', default="constant", type=str, help='loss type')
    
parser.add_argument('--save_dir', default="constant", type=str, help='loss type')
parser.add_argument("--save_embedding", action="store_true", default=False)

parser.add_argument(
    "--train_x",
    action="store_true",
    default=False,
    help="train the exponential map origin",
)
parser.add_argument(
    "--train_c",
    action="store_true",
    default=False,
        help="train the Poincare ball curvature",
)

#######################################################################

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(1)

# mean and standard deviation of channels of iNaturalist images
mean = [0.5055, 0.5269, 0.4941]
std = [0.2138, 0.2125, 0.2598]

# train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
#                                trn.ToTensor(), trn.Normalize(mean, std)])

# test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if args.dataset == 'iNat':
    inat_birds = INat_Birds('dataset/')
    train_data = inat_birds.train_data
    test_data = inat_birds.test_data
    num_classes = len(inat_birds.dataset.classes)

calib_indicator = ''
if args.calibration:
    train_data, val_data = validation_split(train_data, val_share=0.1)
    calib_indicator = '_calib'

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)


# Create model

######################################################################

if args.use_hyperbolic:
    args.save_model_dir = "./saved_models/hyperbolic/"

    net = WideResNet_Hyperbolic(args.layers, num_classes, args.widen_factor, dropRate=args.droprate, args=args)

else:
    args.save_model_dir = "./saved_models/euclidean/"

    net = resnet18(num_classes=num_classes)

######################################################################

print (net)

start_epoch = 0

# Restore model if desired
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        model_name = os.path.join(args.load, args.dataset + calib_indicator + '_' + args.model +
                                  '_baseline_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders


sgd = torch.optim.SGD(
        net.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)

######################################################################

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler_sgd = torch.optim.lr_scheduler.LambdaLR(
    sgd,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))


# /////////////// Training ///////////////        
def train():
    net.train()  # enter train mode
    
    feature_embedding = []

    loss_avg = 0.0
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        
        # forward
        if args.use_hyperbolic:

            # compute output
            if epoch == args.epochs-1:
                x = net(data, True, True)
            else:
                x = net(data, True, False)
        else:

            x = net(data)

        # backward

        sgd.zero_grad()
        
        loss = F.cross_entropy(x, target)
        loss.backward()
        
        sgd.step()

        scheduler_sgd.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg

# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            
            if args.use_hyperbolic:

                output = net(data, False, False)
            else:
                output = net(data)

            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


if args.test:
    test()
    print(state)
    exit()

# Make save directory

print('Beginning Training\n')

# Main loop
for epoch in range(start_epoch, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()

    test()

    if args.use_hyperbolic:
        
        prefix = args.save_model_dir + args.dataset + calib_indicator + '_' + args.model + '_baseline_epoch_' + str(epoch) + "_max_clip_norm_" + str(args.max_clip_norm) + "_" + str(args.dim) + 'D_' +  "curvature_" + str(args.c)


        path_name = prefix + '.pt'

        # Save model
        torch.save(net.state_dict(), os.path.join(path_name))
        
        # Let us not waste space and delete the previous model
        prefix = args.save_model_dir + args.dataset + calib_indicator + '_' + args.model + '_baseline_epoch_' + str(epoch-1) + "_max_clip_norm_" + str(args.max_clip_norm) + "_" + str(args.dim) + 'D_' +  "curvature_" + str(args.c)


        prev_path = prefix + '.pt'
        
    else:

         # Save model
        torch.save(net.state_dict(),
                   os.path.join(args.save_model_dir  + args.dataset + calib_indicator + '_' + args.model + '_baseline_epoch_' + str(epoch) + '.pt'))
        
        # Let us not waste space and delete the previous model
        prev_path = os.path.join(args.save_model_dir +  args.dataset + calib_indicator + '_' + args.model + '_baseline_epoch_' + str(epoch-1) + '.pt')
               

    if os.path.exists(prev_path): os.remove(prev_path)

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy'])
    )