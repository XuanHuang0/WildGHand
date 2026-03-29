import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle
import random
import torch.nn.functional as F



from utils.lr_sc import StepLR_withWarmUp
from utils.DataProvider import DataProvider
from utils.utils import get_mano_path

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
SYN_BN = True

def freeze_model(model):
    for (name, params) in model.named_parameters():
        params.requires_grad = False


def set_bn_momentum(module, momentum=0.01):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momentum
    else:
        for name, child in module.named_children():
            set_bn_momentum(child, momentum=momentum)


def freeze_bn(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()
    else:
        for name, child in module.named_children():
            freeze_bn(child)


def clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device

    if norm_type == torch.inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        # print("detect nan" if total_norm.isnan() else "detect inf")

        for p in parameters:
            p_grad_ = p.grad.detach()
            nan_idxs = torch.isnan(p_grad_)
            inf_idxs = torch.isinf(p_grad_)
            p_grad_[nan_idxs] = 0
            p_grad_[inf_idxs] = 0
            # if nan_idxs is not None or inf_idxs is not None:
            #     print(p_grad_.shape)
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite)

    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))
    return total_norm


def train_gcn(rank=0, world_size=1, cfg=None, dist_training=False, annos=None, tcfg=None):
    if dist_training:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(cfg.TRAIN.DIST_PORT)
        print("Init distributed training on local rank {}".format(rank))
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if hasattr(cfg.INTHAND, 'view_num'):
        view_num = cfg.INTHAND.view_num
    else:
        view_num = 1

    kd_loss_weight = 0 if not hasattr(cfg.INTHAND, 'kd_loss_weight') else cfg.INTHAND.kd_loss_weight

    if cfg.INTHAND.decoder3:
        from core.mLoss import GraphLoss, calc_loss_GCN
        from models.tmodel import load_model
        from models.emodel import load_model as stu_load_model
    else:
        from models.model import load_model as stu_load_model
        from core.Loss import GraphLoss, calc_loss_GCN

    from core.loader_lmdb_mv import handDataset

    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    mano_path = get_mano_path()

    if kd_loss_weight > 0 and tcfg is not None:
        teacher_net = load_model(tcfg)
        pytorch_total_params = sum(p.numel() for p in teacher_net.parameters() if p.requires_grad)
        print("number of total param: {}".format(pytorch_total_params / 1e6))

        state = torch.load(tcfg.tmodel, map_location='cpu')
        state2 = {}
        for k, v in state.items():
            if k.startswith('module'):
                k = k[7:]
            state2[k] = v
        teacher_net.load_state_dict(state2, strict=True)
        teacher_net.to(rank)
        teacher_net.eval()


    # -------------------------------------------------
    # | 1. load model/optimizer/scheduler/tensorboard |
    # -------------------------------------------------
    # load network
    network = stu_load_model(cfg)
    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print("number of total param: {}".format(pytorch_total_params / 1e6))
    pretrained = True
    new_encoder = True
    part_cls = False if not hasattr(cfg.INTHAND, 'part_cls') else cfg.INTHAND.part_cls
    aux_size = 64 if not hasattr(cfg.INTHAND, 'aux_size') else cfg.INTHAND.aux_size


    if part_cls:
        print("part classification")
    if pretrained and (cfg.MODEL_PARAM.MODEL_PRETRAIN_PATH == "none") and cfg.MODEL.ENCODER_TYPE == "resnet50":
        cpkt = torch.load(os.path.join('misc', 'model', 'pre_encdec.pth'), map_location='cpu')
        if new_encoder:
            cpkt2 = {}
            for k, v in cpkt.items():
                if k.startswith('encoder.resnet'):
                    cpkt2[k] = v
            network.load_state_dict(cpkt2, strict=False)
            print("load pretrained intag encoder_resnet {} items".format(len(cpkt2.keys())))
        else:
            network.load_state_dict(cpkt, strict=False)
            print("load pretrained intag encoder decoder")
        resume_training = False
    else:
        if cfg.MODEL_PARAM.MODEL_PRETRAIN_PATH == "none":
            resume_training = False
        else:
            resume_training = True

    decoder_from_teacher = False if not hasattr(cfg.INTHAND, 'DECODER_FROM_TEACHER') else cfg.INTHAND.DECODER_FROM_TEACHER

    network.to(rank)
    if decoder_from_teacher and kd_loss_weight > 0:
        # if cfg.MODEL.ENCODER_TYPE == tcfg.MODEL.ENCODER_TYPE:
        network.encoder.geo_decoder.cat_convs.load_state_dict(teacher_net.encoder.geo_decoder.cat_convs.state_dict())
        network.encoder.geo_decoder.up_convs.load_state_dict(teacher_net.encoder.geo_decoder.up_convs.state_dict())
        network.encoder.geo_decoder.x_merge_pyramid.load_state_dict(teacher_net.encoder.geo_decoder.x_merge_pyramid.state_dict())
        network.encoder.geo_decoder.final_layer.load_state_dict(teacher_net.encoder.geo_decoder.final_layer.state_dict())
        network.encoder.geo_decoder.x_merge_out.load_state_dict(teacher_net.encoder.geo_decoder.x_merge_out.state_dict())
        if network.decoder.use_mano and teacher_net.decoder.use_mano:
            network.decoder.mano_head.load_state_dict(teacher_net.decoder.mano_head.state_dict())
        print("load pretrained decoder from teacher network")

    # for pi, (paran, pv) in enumerate(network.named_parameters()):
    #     print("{} {}".format(pi, paran))
    # exit(0)
    if cfg.MODEL.freeze_upsample:
        if hasattr(network.decoder, 'unsample_layer'):
            print("freeze unsample_layer")
            freeze_model(network.decoder.unsample_layer)

    converter = {}
    for hand_type in ['left', 'right']:
        converter[hand_type] = network.decoder.converter[hand_type]

    if dist_training:
        if SYN_BN:
            network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
            print('convert BN to SyncBN')
        network = DDP(
            network, device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
            broadcast_buffers=True,
        )
        if kd_loss_weight > 0:
            if SYN_BN:
                teacher_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_net)
                print('convert BN to SyncBN')
            teacher_net = DDP(
                teacher_net, device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True,
                broadcast_buffers=True,
            )


    # load optimizer
    optim_params = list(filter(lambda p: p.requires_grad, network.parameters()))
    if cfg.TRAIN.OPTIM == 'adam':
        if dist_training:
            optimizer = ZeroRedundancyOptimizer(
                optim_params,
                optimizer_class=torch.optim.Adam,
                lr=cfg.TRAIN.LR
            )
        else:
            optimizer = torch.optim.Adam(optim_params, lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIM == 'rms':
        if dist_training:
            optimizer = ZeroRedundancyOptimizer(
                optim_params,
                optimizer_class=torch.optim.RMSprop,
                lr=cfg.TRAIN.LR
            )
        else:
            optimizer = torch.optim.RMSprop(optim_params, lr=cfg.TRAIN.LR)
    else:
        raise ValueError('wrong optimizer type')

    if resume_training:
        epoch_num = cfg.MODEL_PARAM.MODEL_PRETRAIN_PATH
        epoch_num = int(epoch_num.split('/')[-1].replace(".pth", ""))
        current_epoch = epoch_num
    else:
        current_epoch = cfg.TRAIN.current_epoch

    # load learning rate scheduler
    lr_scheduler = StepLR_withWarmUp(optimizer,
                                     last_epoch=-1,
                                     init_lr=cfg.TRAIN.LR,
                                     warm_up_epoch=cfg.TRAIN.warm_up,
                                     gamma=cfg.TRAIN.lr_decay_gamma,
                                     step_size=cfg.TRAIN.lr_decay_step,
                                     min_thres=1e-3)
    # print('local rank {}: init lr_scheduler, done'.format(rank))
    # lr_scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    if resume_training:
        lr_scheduler._step_count = current_epoch
        lr_scheduler.step(current_epoch)

    # tensorboard
    if rank == 0:
        writer = SummaryWriter(cfg.TB.SAVE_DIR)

    # --------------------------
    # | 2. load dataset & Loss |
    # --------------------------
    trainDataset = handDataset(data_path=cfg.DATASET.INTERHAND_PATH,
                               theta=[-cfg.DATA_AUGMENT.THETA, cfg.DATA_AUGMENT.THETA],
                               scale=[1 - cfg.DATA_AUGMENT.SCALE, 1 + cfg.DATA_AUGMENT.SCALE],
                               uv=[-cfg.DATA_AUGMENT.UV, cfg.DATA_AUGMENT.UV],
                               aux_size=aux_size,
                               train_ratio=cfg.INTHAND.train_ratio,
                               annos=annos,
                               view_num=view_num,
                               part_cls=part_cls,
                               )
    # if dist_training:
    #     sampler = DistributedSampler(trainDataset, shuffle=True, drop_last=True)
    #     provider_train = DataLoader(trainDataset, batch_size=cfg.TRAIN.BATCH_SIZE, sampler=sampler,
    #                             num_workers=4, drop_last=True, pin_memory=True)
    # else:
    #     provider_train = DataLoader(trainDataset, batch_size=cfg.TRAIN.BATCH_SIZE,
    #                             num_workers=world_size, drop_last=True, pin_memory=True)
    # train_batch_per_epoch = len(provider_train)
    provider_train = DataProvider(dataset=trainDataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                                  num_workers=4, dist=dist_training, epoch=current_epoch)
    train_batch_per_epoch = provider_train.batch_per_epoch
    print('local rank {}: init data loader, done, train_batch_per_epcoh: {}'.format(rank, train_batch_per_epoch))

    Loss = {}
    for hand_type in ['left', 'right']:
        with open(mano_path[hand_type], 'rb') as file:
            manoData = pickle.load(file, encoding='latin1')
        Loss[hand_type] = GraphLoss(manoData['f'], level=4, device=rank)
        # device='cuda:{}'.format(rank))

    # print('local rank {}: init training loss, done'.format(rank))

    # ------------
    # | 3. train |
    # ------------
    # print('local rank {}: strat training'.format(rank))
    skip_start = False
    LOSS_THRESH = 200
    thresh_num = 1000
    thresh_count = 0
    act_dict = False if not hasattr(cfg.INTHAND, 'act_dict') else cfg.INTHAND.act_dict

    # for name, child in network.named_modules():
    #     if 'stage' in name:
    #         freeze_bn(child)
    #         print("freeze bn in {}".format(name))

    loss_count = 0
    optimizer.zero_grad()

    scaler = torch.cuda.amp.GradScaler(growth_interval=100)

    for epoch in range(current_epoch, cfg.TRAIN.EPOCHS):
        # if epoch > 30 and act_dict:
        #     network.module.decoder.vol_decoder.reproj3d = True
        broken_count = 0
        network.train()
        train_bar = range(train_batch_per_epoch)
        if rank == 0:
            train_bar = tqdm(train_bar)
        lr_bks = []
        for bIdx in train_bar:
            with torch.no_grad():
                bk_params = {}
                for k, v in network.state_dict().items():
                    bk_params[k] = v.clone()

            total_idx = epoch * train_batch_per_epoch + bIdx

            # ------------
            # | training |
            # ------------

            try:
            #     # may caused by pca, reading errors, etc.
                imgTensors, targets = provider_train.next()
                bs, bvc, bh, bw = imgTensors.shape[:]
                imgTensors = imgTensors.reshape(bs * view_num, -1, bh, bw)
                for tti in range(len(targets)):
                    tshape = []
                    tshape.append(targets[tti].shape[0] * view_num)
                    tshape.append(targets[tti].shape[1] // view_num)
                    for ts in targets[tti].shape[2:]:
                        tshape.append(ts)
                    targets[tti] = targets[tti].reshape(tshape)
            except:
                continue
            # imgTensors, targets = provider_train.next()
            # imgTensors = imgTensors.cuda()

            # if torch.isnan(imgTensors).float().mean()> 0:
            #     print("img")
            #     exit(0)
            # for ti, tk in enumerate(targets):
            #     if torch.isnan(tk).float().mean() > 0:
            #         print(ti)
            #         exit(0)


            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result, paramsDict, handDictList, otherInfo, img_f, img_fmap, img_fmaps = network(imgTensors, mv=view_num)

                if kd_loss_weight > 0:
                    with torch.no_grad():
                        timg_fmaps, timg_fmap_merge = teacher_net(imgTensors)

                    if img_fmap.shape[1] == timg_fmap_merge.shape[1]:
                        kd_loss = F.mse_loss(img_fmap, timg_fmap_merge)
                    else:
                        kd_loss = 0
                    for tei, (tfmap, fmap) in enumerate(zip(timg_fmaps, img_fmaps)):
                        if fmap.shape[1] == tfmap.shape[1]:
                            if fmap.shape[2] != tfmap.shape[2]:
                                fmap = F.interpolate(fmap, size=(tfmap.shape[2], tfmap.shape[3]), mode="bilinear", align_corners=True)
                            kd_loss = kd_loss + F.mse_loss(fmap, tfmap)
                else:
                    kd_loss = 0


                if cfg.MODEL.freeze_upsample:
                    upsample_weight = None
                else:
                    if dist_training:
                        upsample_weight = network.module.decoder.get_upsample_weight()
                    else:
                        upsample_weight = network.decoder.get_upsample_weight()
                if cfg.INTHAND.decoder3:
                    loss, aux_lost_dict, mano_loss_dict, coarsen_loss_dict = \
                        calc_loss_GCN(cfg, epoch,
                                      Loss['left'], Loss['right'],
                                      converter['left'], converter['right'],
                                      result, paramsDict, handDictList, otherInfo,
                                      targets, upsample_weight=upsample_weight, part_cls=part_cls)
                else:
                    loss, aux_lost_dict, mano_loss_dict, coarsen_loss_dict = \
                        calc_loss_GCN(cfg, epoch,
                                      Loss['left'], Loss['right'],
                                      converter['left'], converter['right'],
                                      result, paramsDict, handDictList, otherInfo,
                                      targets, upsample_weight=upsample_weight)


                loss_item = loss.item()

                if kd_loss != 0:
                    loss = loss + kd_loss * (kd_loss_weight * loss_item / (1 + kd_loss.item()))


            if skip_start and loss_item > LOSS_THRESH or loss.isnan() or loss.isinf():
                # if epoch > 1 and loss_item > 200:
                #     # there are some strange loss pikes, ignore them
                for g in optimizer.param_groups:
                    lr_bks.append(g['lr'])
                    g['lr'] = g['lr'] * 1e-2

            else:
                if not skip_start and loss_item < LOSS_THRESH:
                    thresh_count = thresh_count + 1
                    if thresh_count >= thresh_num:
                        skip_start = True


            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(network.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            isnan = torch.stack([torch.isnan(p).any() or torch.isinf(p).any() for p in network.parameters()]).any()
            isnan = torch.logical_or(isnan, loss.isnan() or loss.isinf())
            if isnan:
                broken_count = broken_count + 1
                with torch.no_grad():
                    network.load_state_dict(bk_params)

            if len(lr_bks) > 0:
                for g, lr in zip(optimizer.param_groups, lr_bks):
                    g['lr'] = lr
                lr_bks = []

            # --------
            # | tqdm |
            # --------
            if rank == 0:
                train_bar.set_description('train, epoch:{}'.format(epoch))
                train_bar.set_postfix({'totalLoss': loss_item, 'kdLoss': kd_loss.item() if kd_loss !=0 else 0, 'broken_times': broken_count})

            # ---------------
            # | tensorboard |
            # ---------------
            if rank == 0:
                writer.add_scalar('learning_rate', lr_scheduler.get_lr()[0], total_idx)
                writer.add_scalar('train/total_loss', loss_item, total_idx)
                # if epoch > 1 and loss.item() > 500:
                #     torch.save({'imgTesnors': imgTensors, 'targets': targets}, 'error500.pth')
                for k, v in mano_loss_dict.items():
                    if k != 'total_loss':
                        writer.add_scalar('train/mano_{}'.format(k), v.mean().item(), total_idx)
                for k, v in aux_lost_dict.items():
                    if k != 'total_loss':
                        writer.add_scalar('train/aux_{}'.format(k), v.mean().item(), total_idx)
                for k, v in coarsen_loss_dict.items():
                    if k != 'total_loss':
                        for t in range(len(v)):
                            writer.add_scalar('train/coarsen_{}_{}'.format(k, t), v[t].mean().item(), total_idx)

            "takes about 2 hours"
            if train_batch_per_epoch > 5000 and (bIdx + 1) % 2500 == 0:
                if rank == 0:
                    torch.save(network.state_dict(), os.path.join(cfg.SAVE.SAVE_DIR, str(epoch + 1) + '.pth'))

        lr_scheduler.step()
        if (epoch + 1) % cfg.SAVE.SAVE_GAP == 0:
            if rank == 0:
                torch.save(network.state_dict(), os.path.join(cfg.SAVE.SAVE_DIR, str(epoch + 1) + '.pth'))

