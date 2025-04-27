import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

# Adjust these imports to match your code organization
from .data import CIFData, CIFDataFeaturizer
from .data import collate_pool, get_train_val_test_loader
from .model import CrystalGraphConvNet
from .util import save_checkpoint, AverageMeter, class_eval, mae, Normalizer
from .model_sph_harmonics import SpookyModel, SpookyModelVectorized


parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, starting with the path to root dir, '
                         'then other options')
parser.add_argument('--task', choices=['regression', 'classification','Fxyz'],
                    default='regression',
                    help='complete a regression or classification task '
                         '(default: regression). (Fxyz: do regression with forces)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA (run on CPU only).')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--lr-milestones', default=[1000000], nargs='+', type=int,
                    metavar='N', help='scheduler milestones (default: [1000000])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for SGD')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                    help='path to pretrained model (default: none)')
parser.add_argument('--all-elems', nargs='*', type=int, default=[0],
                    help='If training an IAP, specify all possible atom types that appear')
parser.add_argument('--start-fine-tuning-epoch', type=int, default=0,
                    help='Epoch at which to start fine-tuning')

train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                        help='fraction of data for training (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                        help='number of training data (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                        help='fraction of data for val (default 0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of val data (default none)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                        help='fraction of data for test (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data (default none)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')
parser.add_argument('--freeze-embedding', action='store_true',
                    help='Freeze the embedding layer')
parser.add_argument('--freeze-conv', action='store_true',
                    help='Freeze the convolutional layer')
parser.add_argument('--freeze-fc', type=int, default=0,
                    help='Number of fully-connected layers to freeze')
parser.add_argument('--fine-tune', action='store_true',
                    help='Perform additional fine-tuning after initial transfer learning')
parser.add_argument("--seed", default=0, type=int,
                    help='PyTorch seed for reproducibility')
parser.add_argument('--cross-validation', default=None, type=str, metavar='N',
                    help='type of cross validation to be used')
parser.add_argument('--cross-param', default=None, type=int, metavar='N',
                    help='parameter for certain cross validation methods')
parser.add_argument('--counter', default=0, type=int, metavar='N',
                    help='iteration of cross validation method')

# Additional user options
parser.add_argument('--resultdir', default='.', type=str, metavar='N',
                    help='directory to write test results')
parser.add_argument('--jit', action='store_true',
                    help='Create a serialized Torch script')
parser.add_argument('--crys-spec', default=None, type=str, metavar='N',
                    help='ext for global crystal features (e.g. example.ext)')
parser.add_argument('--atom-spec', default=None, type=str, metavar='N',
                    help='ext for per-atom features (e.g. example.ext)')
parser.add_argument('--csv-ext', default='', type=str,
                    help='append to id_prop.csv, e.g. id_prop.csv.split1 => csv_ext=.split1')
parser.add_argument('--model-type', default='cgcnn', type=str,
                    choices=['cgcnn','spooky'],
                    help='Which model to use (CGCNN or Spooky)')
parser.add_argument('--njmax', default=75, type=int,
                    help='Max #neighbors for spherical harmonics featurization')
parser.add_argument('--init-embed-file', default='atom_init.json', type=str,
                    help='file for the initial atom embeddings')

args = parser.parse_args(sys.argv[1:])

# Save training arguments
with open(os.path.join(args.resultdir, 'parameters_CGCNNtrain.json'), 'w') as fp:
    json.dump(vars(args), fp)

# Determine if we can use CUDA
args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.

###############################
# Helper function to fix device mismatch
###############################
def move_to_device(obj, device):
    """
    Recursively move Tensors (and lists/tuples of Tensors) to the given device.
    This ensures that any indexing Tensors like crystal_atom_idx
    also end up on the correct device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    else:
        return obj


def main():
    global args, best_mae_error

    # Set CPU/GPU seeds for reproducibility
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Create device object
    device = torch.device('cuda' if args.cuda else 'cpu')

    # Print device info
    print(f"[INFO] Using device: {device}")
    if device.type == 'cuda':
        print(f"[INFO] GPU Device Name: {torch.cuda.get_device_name(0)}")

    # open log file
    f = open(os.path.join(args.resultdir, "train.log"), "w")

    # load dataset
    print(args.task)
    dataset = CIFData(*args.data_options,
                      args.task=='Fxyz',
                      args.all_elems,
                      crys_spec=args.crys_spec,
                      atom_spec=args.atom_spec,
                      csv_ext=args.csv_ext,
                      model_type=args.model_type,
                      njmax=args.njmax,
                      init_embed_file=args.init_embed_file)

    collate_fn = collate_pool

    # If cross-validation is k-fold/bootstrapping/etc., do not return test set
    return_test = True
    if args.cross_validation in [
        'k-fold', 'k-fold-cross-validation', 'bootstrapping', 'bootstrap',
        'leave-p-out', 'leave-one-out'
    ]:
        return_test = False

    # Create data loaders
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=return_test,
        cross_validation=args.cross_validation,
        cross_param=args.cross_param,
        counter=args.counter,
        random_seed=args.seed
    )

    # Build normalizers
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
        normalizer_Fxyz = None
    else:
        if len(dataset) < 500:
            warnings.warn('Dataset has fewer than 500 data points. Lower accuracy is expected.')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]
        _, sample_target, sample_target_Fxyz, _ = collate_fn(sample_data_list)
        normalizer = Normalizer(sample_target)
        if args.task == 'Fxyz':
            raise NotImplementedError("Forces not implemented yet.")
        else:
            normalizer_Fxyz = Normalizer(sample_target)

    # Build the model
    structures, _, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    global_fea_len = len(structures[7]) if args.crys_spec is not None else 0

    if args.model_type == 'cgcnn':
        model = CrystalGraphConvNet(
            orig_atom_fea_len, nbr_fea_len,
            atom_fea_len=args.atom_fea_len,
            n_conv=args.n_conv,
            h_fea_len=args.h_fea_len,
            n_h=args.n_h,
            classification=(args.task=='classification'),
            Fxyz=(args.task=='Fxyz'),
            all_elems=args.all_elems,
            global_fea_len=global_fea_len
        )
    elif args.model_type == 'spooky':
        if args.njmax > 0:
            model = SpookyModelVectorized(
                orig_atom_fea_len,
                atom_fea_len=args.atom_fea_len,
                n_conv=args.n_conv,
                h_fea_len=args.h_fea_len,
                n_h=args.n_h,
                global_fea_len=global_fea_len,
                njmax=args.njmax
            )
        else:
            model = SpookyModel(
                orig_atom_fea_len,
                atom_fea_len=args.atom_fea_len,
                n_conv=args.n_conv,
                h_fea_len=args.h_fea_len,
                n_h=args.n_h,
                global_fea_len=global_fea_len
            )

    # Print param count
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable params: {num_trainable}")

    # Send model to GPU if available
    model.to(device)

    # Possibly freeze layers
    if args.freeze_embedding and hasattr(model, 'embedding'):
        for param in model.embedding.parameters():
            param.requires_grad = False
    if args.freeze_conv and hasattr(model, 'convs'):
        for conv_layer in model.convs:
            for param in conv_layer.parameters():
                param.requires_grad = False
    if args.freeze_fc > 0 and hasattr(model, 'fcs'):
        for i in range(args.freeze_fc):
            for param in model.fcs[i].parameters():
                param.requires_grad = False

    criterion = nn.NLLLoss() if args.task == 'classification' else nn.MSELoss()

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    if path := args.resume or args.pretrain:
        if os.path.isfile(path):
            print(f"=> loading checkpoint '{path}'")
            checkpoint = torch.load(path, map_location='cpu')
            model_args = argparse.Namespace(**checkpoint['args'])

            def assert_same_params(arg):
                model_arg = vars(model_args)[arg]
                this_arg = vars(args)[arg]
                if model_arg != this_arg:
                    print(f'Error: checkpoint param {arg}={model_arg} while command line arg={this_arg}.')
                    sys.exit(1)

            assert_same_params('atom_fea_len')
            assert_same_params('n_conv')
            assert_same_params('h_fea_len')
            assert_same_params('n_h')

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])

            if args.resume:
                args.start_epoch = checkpoint['epoch']
                best_mae_error = checkpoint['best_mae_error']
            else:
                for param in model.parameters():
                    param.requires_grad = True
                if hasattr(model, 'embedding'):
                    for param in model.embedding.parameters():
                        param.requires_grad = False
            print(f"=> loaded checkpoint '{path}' (epoch {checkpoint['epoch']})")

    unfreeze_epochs = [int(0.3 * args.epochs), int(0.6 * args.epochs), int(0.7 * args.epochs)]

    train_start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        # Possibly unfreeze layers
        if epoch in unfreeze_epochs:
            if epoch == unfreeze_epochs[0]:
                if hasattr(model, 'fcs') and len(model.fcs) > 0:
                    for param in model.fcs[-1].parameters():
                        param.requires_grad = True
                print("Unfroze the last fully connected layer.")
            elif epoch == unfreeze_epochs[1]:
                if hasattr(model, 'convs') and len(model.convs) > 0:
                    for param in model.convs[-1].parameters():
                        param.requires_grad = True
                print("Unfroze the last convolutional layer.")
            elif epoch == unfreeze_epochs[2]:
                for param in model.parameters():
                    param.requires_grad = True
                print("Fully unfroze the model.")

        summary = train(train_loader, model, criterion, optimizer, epoch,
                        normalizer, None, device)
        f.write(summary)

        mae_error, summary = validate(val_loader, model, criterion,
                                      normalizer, None, device=device)
        f.write(summary)

        import math
        if math.isnan(mae_error):
            print('Exit due to NaN in validation')
            sys.exit(1)

        scheduler.step()

        is_best = (mae_error < best_mae_error if args.task in ['regression', 'Fxyz']
                   else mae_error > best_mae_error)
        best_mae_error = min(mae_error, best_mae_error) if is_best else best_mae_error
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'normalizer_Fxyz': None,
            'args': vars(args)
        }, is_best, args.resultdir)

    total_train_time = time.time() - train_start_time
    print(f"[INFO] Training completed in {total_train_time:.2f} seconds")

    print('---------Evaluate Model on Test Set---------------')
    best_ckpt = torch.load(os.path.join(args.resultdir, 'model_best.pth.tar'), map_location='cpu')
    model.load_state_dict(best_ckpt['state_dict'])
    model.to(device)

    mae, summary = validate(test_loader, model, criterion, normalizer, None,
                            device=device, test=True)

    if args.jit:
        sm = torch.jit.script(model)
        sm.save(os.path.join(args.resultdir, "model_best.pt"))
        print("TorchScript model saved as 'model_best.pt'")

    f.write('---------Evaluate Model on Test Set---------------\n')
    f.write(summary)
    f.close()


def train(train_loader, model, criterion, optimizer, epoch,
          normalizer, normalizer_Fxyz, device):
    summary = ""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    if args.task == 'regression':
        mae_errors = AverageMeter()
        mae_Fxyz_errors = None
    elif args.task == 'Fxyz':
        mae_errors = AverageMeter()
        mae_Fxyz_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    model.train()
    end = time.time()

    for i, (inputs, target, target_Fxyz, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        #############################################
        # The fix: Move everything recursively to GPU
        #############################################
        inputs = move_to_device(inputs, device)
        target = move_to_device(target, device)
        target_Fxyz = move_to_device(target_Fxyz, device)
        #############################################

        if args.model_type == 'cgcnn':
            input_var = (
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6],
                inputs[7],
                inputs[8]
            )
        elif args.model_type == 'spooky':
            input_var = (
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                inputs[6]
            )

        # Possibly compute repulsive energy
        if args.all_elems != [0] and hasattr(model, 'compute_repulsive_ener'):
            if args.model_type == 'cgcnn':
                crys_rep_ener = model.compute_repulsive_ener(
                    input_var[3],
                    input_var[4],
                    input_var[5],
                    input_var[6]
                )
            else:
                crys_rep_ener = torch.zeros_like(target)
        else:
            crys_rep_ener = torch.zeros_like(target)

        if args.task == 'regression':
            target_normed = normalizer.norm(target - crys_rep_ener)
        elif args.task == 'Fxyz':
            target_normed = normalizer.norm(target)
            target_Fxyz = torch.flatten(target_Fxyz, start_dim=0, end_dim=1)
            target_Fxyz_normed = normalizer_Fxyz.norm(target_Fxyz)
        else:
            target_normed = target.view(-1).long()

        output = model(*input_var)

        loss_orig = criterion(output[0], target_normed)
        alpha = 1
        if args.task == 'Fxyz':
            loss_Fxyz = criterion(output[1], target_Fxyz_normed)
            loss = loss_orig + alpha * loss_Fxyz
        else:
            loss = loss_orig

        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output[0].detach().cpu()) + crys_rep_ener.detach().cpu(),
                            target.detach().cpu())
            losses.update(loss.item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        elif args.task == 'Fxyz':
            mae_error = mae(normalizer.denorm(output[0].detach().cpu()), target.detach().cpu())
            mae_Fxyz_error = mae(normalizer_Fxyz.denorm(output[1].detach().cpu()), target_Fxyz.detach().cpu())
            losses.update(loss.item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            mae_Fxyz_errors.update(mae_Fxyz_error, target_Fxyz.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(
                output[0].detach().cpu(),
                target.detach().cpu()
            )
            losses.update(loss.item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        optimizer.zero_grad()
        loss.backward()
        if args.fine_tune and epoch >= args.start_fine_tuning_epoch:
            for param in model.parameters():
                param.requires_grad = True
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                summary = (
                    f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})"
                )
            elif args.task == 'Fxyz':
                summary = (
                    f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t"
                    f"MAE_Fxyz {mae_Fxyz_errors.val:.3f} ({mae_Fxyz_errors.avg:.3f})"
                )
            else:
                summary = (
                    f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Accu {accuracies.val:.3f} ({accuracies.avg:.3f})\t"
                    f"Precision {precisions.val:.3f} ({precisions.avg:.3f})\t"
                    f"Recall {recalls.val:.3f} ({recalls.avg:.3f})\t"
                    f"F1 {fscores.val:.3f} ({fscores.avg:.3f})\t"
                    f"AUC {auc_scores.val:.3f} ({auc_scores.avg:.3f})"
                )
            print(summary)
    return summary + '\n'


def validate(val_loader, model, criterion, normalizer, normalizer_Fxyz,
             device=None, test=False):
    summary = ""
    batch_time = AverageMeter()
    losses = AverageMeter()

    if args.task == 'regression':
        mae_errors = AverageMeter()
        mae_Fxyz_errors = None
    elif args.task == 'Fxyz':
        mae_errors = AverageMeter()
        mae_Fxyz_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []
        test_targets_Fxyz = []
        test_preds_Fxyz = []

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (inputs, target, target_Fxyz, batch_cif_ids) in enumerate(val_loader):
            #############################################
            # Move inputs, target, Fxyz to device
            #############################################
            inputs = move_to_device(inputs, device)
            target = move_to_device(target, device)
            target_Fxyz = move_to_device(target_Fxyz, device)

            if args.model_type == 'cgcnn':
                input_var = (
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
                    inputs[7],
                    inputs[8]
                )
            elif args.model_type == 'spooky':
                input_var = (
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6]
                )

            if args.all_elems != [0] and hasattr(model, 'compute_repulsive_ener'):
                if args.model_type == 'cgcnn':
                    crys_rep_ener = model.compute_repulsive_ener(
                        input_var[3],
                        input_var[4],
                        input_var[5],
                        input_var[6]
                    )
                else:
                    crys_rep_ener = torch.zeros_like(target)
            else:
                crys_rep_ener = torch.zeros_like(target)

            if args.task == 'regression':
                target_normed = normalizer.norm(target - crys_rep_ener)
            elif args.task =='Fxyz':
                target_normed = normalizer.norm(target)
                target_Fxyz = torch.flatten(target_Fxyz, start_dim=0, end_dim=1)
                target_Fxyz_normed = normalizer_Fxyz.norm(target_Fxyz)
            else:
                target_normed = target.view(-1).long()

            output = model(*input_var)
            loss_orig = criterion(output[0], target_normed)

            alpha = 1
            if args.task == 'Fxyz':
                loss_Fxyz = criterion(output[1], target_Fxyz_normed)
                loss = loss_orig + alpha * loss_Fxyz
            else:
                loss = loss_orig

            if args.task == 'regression':
                mae_error = mae(
                    normalizer.denorm(output[0].cpu()) + crys_rep_ener.cpu(),
                    target.cpu()
                )
                losses.update(loss.item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))

                if test:
                    test_pred = normalizer.denorm(output[0].cpu()) + crys_rep_ener.cpu()
                    test_targets += target.view(-1).cpu().tolist()
                    test_preds += test_pred.view(-1).tolist()
                    test_cif_ids += batch_cif_ids

            elif args.task == 'Fxyz':
                mae_error = mae(normalizer.denorm(output[0].cpu()), target.cpu())
                mae_Fxyz_error = mae(normalizer_Fxyz.denorm(output[1].cpu()), target_Fxyz.cpu())
                losses.update(loss.item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))
                mae_Fxyz_errors.update(mae_Fxyz_error, target_Fxyz.size(0))
                if test:
                    test_pred = normalizer.denorm(output[0].cpu())
                    test_targets += target.view(-1).cpu().tolist()
                    test_preds += test_pred.view(-1).tolist()
                    test_cif_ids += batch_cif_ids

                    test_pred_Fxyz = normalizer_Fxyz.denorm(output[1].cpu())
                    test_targets_Fxyz.append(target_Fxyz.view(-1).cpu().tolist())
                    test_preds_Fxyz.append(test_pred_Fxyz.view(-1).tolist())

            else:
                # classification
                accuracy, precision, recall, fscore, auc_score = class_eval(
                    output[0].cpu(), target.cpu()
                )
                losses.update(loss.item(), target.size(0))
                accuracies.update(accuracy, target.size(0))
                precisions.update(precision, target.size(0))
                recalls.update(recall, target.size(0))
                fscores.update(fscore, target.size(0))
                auc_scores.update(auc_score, target.size(0))

                if test:
                    test_pred = torch.exp(output[0].cpu())
                    test_preds += test_pred[:, 1].tolist()
                    test_targets += target.view(-1).cpu().tolist()
                    test_cif_ids += batch_cif_ids

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                if args.task == 'regression':
                    summary1 = (
                        f"Test: [{i}/{len(val_loader)}]\t"
                        f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                        f"MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})"
                    )
                elif args.task == 'Fxyz':
                    summary1 = (
                        f"Test: [{i}/{len(val_loader)}]\t"
                        f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                        f"MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t"
                        f"MAE_Fxyz {mae_Fxyz_errors.val:.3f} ({mae_Fxyz_errors.avg:.3f})"
                    )
                else:
                    summary1 = (
                        f"Test: [{i}/{len(val_loader)}]\t"
                        f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                        f"Accu {accuracies.val:.3f} ({accuracies.avg:.3f})\t"
                        f"Precision {precisions.val:.3f} ({precisions.avg:.3f})\t"
                        f"Recall {recalls.val:.3f} ({recalls.avg:.3f})\t"
                        f"F1 {fscores.val:.3f} ({fscores.avg:.3f})\t"
                        f"AUC {auc_scores.val:.3f} ({auc_scores.avg:.3f})"
                    )
                print(summary1)
                summary += summary1 + '\n'

    if test:
        star_label = '**'
        import csv
        with open(os.path.join(args.resultdir, 'test_results.csv'), 'w') as f_csv:
            writer = csv.writer(f_csv)
            for cif_id, targ, pred_ in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, targ, pred_))

        if args.task == 'Fxyz':
            with open(os.path.join(args.resultdir,'test_results_Fxyz.csv'), 'w') as f2:
                for cif_id, tgF, pdF in zip(test_cif_ids, test_targets_Fxyz, test_preds_Fxyz):
                    for i_, (tg, pd) in enumerate(zip(tgF, pdF)):
                        f2.write(f"{cif_id},{int(np.floor(i_/3))},{tg:.5f},{pd:.5f}\n")
    else:
        star_label = '*'

    if args.task == 'regression':
        summary1 = f" {star_label} MAE {mae_errors.avg:.3f}"
        print(summary1)
        return mae_errors.avg, summary + summary1 + '\n'
    elif args.task == 'Fxyz':
        summary1 = (f" {star_label} MAE {mae_errors.avg:.3f} "
                    f"MAE_Fxyz {mae_Fxyz_errors.avg:.3f}")
        print(summary1)
        return mae_errors.avg + mae_Fxyz_errors.avg, summary + summary1 + '\n'
    else:
        summary1 = f" {star_label} AUC {auc_scores.avg:.3f}"
        print(summary1)
        return auc_scores.avg, summary + summary1 + '\n'


if __name__ == '__main__':
    main()
