import time
import torch
from torch import nn
import numpy as np
from os.path import join
import time
from pdb import set_trace
import argparse
from loguru import logger
from tqdm import tqdm

import math
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups


def inference_save(inference_loader, model, log, local_rank, repeat_time, save_dir):

    end = time.time()

    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()

    separate_save = False
    if len(inference_loader) > 20000:
        log.info("separate save for large files")
        separate_save = True
        batch_cnt = 0

    with torch.no_grad():
        model.eval()
        submodules = list(model.children())
        vision_transformer = submodules[0]
        dino_head = submodules[1]
        model = nn.Sequential(vision_transformer)
        new_dino_head = nn.Identity()
        model.add_module("dino_head", new_dino_head)
        print(model)


        for cnt_run in range(repeat_time):
            features_all = []
            features_before_proj_all = []
            idxs_all = []
            labels_all = []

            for i, (inputs, labels, idxs, _) in enumerate(tqdm(inference_loader)):
                data_time = time.time() - end
                data_time_meter.update(data_time)

                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                idxs = idxs.cuda(non_blocking=True)

                features_before_proj = model(inputs)
                features = dino_head(features_before_proj)[0]


                if local_rank == 0:
                    features_before_proj_all.append(features_before_proj.detach().cpu())
                    features_all.append(features.detach().cpu())
                    idxs_all.append(idxs.detach().cpu())
                    labels_all.append(labels.detach().cpu())

                train_time = time.time() - end
                end = time.time()
                train_time_meter.update(train_time)

                # torch.cuda.empty_cache()
                if i % 10 == 0 or i == len(inference_loader) - 1:
                    log.info('Run: [{0}][{1}/{2}]\t'
                             'data_time: {data_time.val:.2f} ({data_time.avg:.2f})\t'
                             'train_time: {train_time.val:.2f} ({train_time.avg:.2f})\t'.format(
                        cnt_run, i, len(inference_loader),
                        data_time=data_time_meter, train_time=train_time_meter))

                if separate_save and len(idxs_all) >= 10000:
                    features_all = torch.cat(features_all, dim=0).cpu()
                    features_before_proj_all = torch.cat(features_before_proj_all, dim=0).cpu()
                    idxs_all = torch.cat(idxs_all, dim=0).cpu().numpy()
                    labels_all = torch.cat(labels_all, dim=0).cpu().numpy()

                    torch.save(features_all, join(save_dir, "features_all_time{}_batch{}.pt".format(cnt_run, batch_cnt)))
                    torch.save(features_before_proj_all,
                               join(save_dir, "features_before_proj_all_time{}_batch{}.pt".format(cnt_run, batch_cnt)))
                    np.save(join(save_dir, "idxs_all_time{}_batch{}".format(cnt_run, batch_cnt)), idxs_all)
                    np.save(join(save_dir, "labels_all_time{}_batch{}".format(cnt_run, batch_cnt)), labels_all)

                    batch_cnt += 1
                    features_all = []
                    features_before_proj_all = []
                    idxs_all = []
                    labels_all = []

            # calculate prob
            features_all = torch.cat(features_all, dim=0).cpu()
            features_before_proj_all = torch.cat(features_before_proj_all, dim=0).cpu()
            idxs_all = torch.cat(idxs_all, dim=0).cpu().numpy()
            labels_all = torch.cat(labels_all, dim=0).cpu().numpy()

            torch.save(features_all, join(save_dir, "features_all_time{}.pt".format(cnt_run)))
            torch.save(features_before_proj_all, join(save_dir, "features_before_proj_all_time{}.pt".format(cnt_run)))
            np.save(join(save_dir, "idxs_all_time{}".format(cnt_run)), idxs_all)
            np.save(join(save_dir, "labels_all_time{}".format(cnt_run)), labels_all)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars',
                        help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)

    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float,
                        help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float,
                        help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--imb_ratio', default=100, type=int)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')

    torch.backends.cudnn.benchmark = True

    args.interpolation = 3
    args.crop_pct = 0.875

    # 数据集载入
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         test_transform,
                                                                                         test_transform,
                                                                                         args)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    # 模型权重载入
    dir = '/checkpoints/model_best.pt'
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    print(f'Using weights from...')
    state_dict_old = torch.load(dir)['model']
    state_dict = {}
    state_dict_p = {}
    for k, v in state_dict_old.items():
        if k[0] == '1':
            state_dict_p[k[2:]] = v
        else:
            state_dict[k[2:]] = v

    backbone.load_state_dict(state_dict)

    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    projector.load_state_dict(state_dict_p)

    model = nn.Sequential(backbone, projector)
    model = model.to('cuda:0')

    logger.add('test_result/log_0.txt')
    inference_save(train_loader, model, logger, 0, 5, 'test_result')
