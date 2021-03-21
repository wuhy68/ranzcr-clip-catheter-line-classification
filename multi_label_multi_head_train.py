import argparse
import pandas as pd
import numpy as np
import typing as tp
import os
import time
import cv2
import random
from collections import OrderedDict

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast as autocast
import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist

from warmup_scheduler import GradualWarmupScheduler
from sklearn.metrics import roc_auc_score

import albumentations
from albumentations import *

import timm
from functools import reduce

from warnings import filterwarnings
filterwarnings("ignore")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # for faster training, but not deterministic

def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt

def gather_tensor(tensor):
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list)

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

class NIHDataset(Dataset):
    def __init__(self, df, target_cols, transform=None):
        self.df = df
        self.file_names = df['StudyInstanceUID'].values
        self.transform = transform
        self.labels = df[target_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{NIH_PATH}/{file_name}.png'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)
        label = torch.tensor(self.labels[idx]).float()

        return torch.tensor(image).float(), label

class RANZCRDataset(Dataset):
    def __init__(self, df, mode, target_cols, transform=None):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.labels = df[target_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        img = cv2.imread(row.file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']

        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        label = torch.tensor(self.labels[index]).float()
        if self.mode == 'test':
            return torch.tensor(img).float()
        else:
            return torch.tensor(img).float(), label

class OrderedDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

def macro_multilabel_auc(label, pred, target_cols):
    aucs = []
    # print(label)
    for i in range(len(target_cols)):
        aucs.append(roc_auc_score(label[:, i], pred[:, i]))
    return aucs

def load_model(model_name, checkpoint):
    state_dict = torch.load(checkpoint)["model"]  # 模型可以保存为pth文件，也可以为pt文件。

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[19:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。

    model = timm.create_model(model_name, pretrained=False, num_classes=11)
    model.load_state_dict(new_state_dict)

    return model

def get_activation(activ_name: str = "relu"):
    """"""
    act_dict = {
        "relu": nn.ReLU(inplace=True),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "identity": nn.Identity(),
        "gelu": nn.GELU()
    }
    if activ_name in act_dict:
        return act_dict[activ_name]
    else:
        raise NotImplementedError

class Conv2dBNActiv(nn.Module):
    """Conv2d -> (BN ->) -> Activation"""

    def __init__(
            self, in_channels: int, out_channels: int,
            kernel_size: int, stride: int = 1, padding: int = 0,
            bias: bool = False, use_bn: bool = True, activ: str = "relu"
    ):
        """"""
        super(Conv2dBNActiv, self).__init__()
        layers = []
        layers.append(nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding, bias=bias))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(get_activation(activ))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward"""
        return self.layers(x)

class MultiHeadModel(nn.Module):

    def __init__(
            self, base_name: str = 'resnext50_32x4d',
            out_dims_head: tp.List[int] = [3, 4, 3, 1], pretrained=False):
        """"""
        self.base_name = base_name
        self.n_heads = len(out_dims_head)
        super(MultiHeadModel, self).__init__()

        # # load base model
        base_model = timm.create_model(base_name, pretrained=pretrained, num_classes=21843)
        in_features = base_model.num_features

        # # remove global pooling and head classifier
        base_model.reset_classifier(0, '')

        # # Shared CNN Bacbone
        self.backbone = base_model

        # # Multi Heads.
        for i, out_dim in enumerate(out_dims_head):
            layer_name = f"head_{i}"
            layer = nn.Sequential(
                SpatialAttentionBlock(in_features, [64, 32, 16, 1]),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, in_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(in_features, out_dim))
            setattr(self, layer_name, layer)

    def forward(self, x):
        """"""
        h = self.backbone(x)
        hs = [
            getattr(self, f"head_{i}")(h) for i in range(self.n_heads)]
        y = torch.cat(hs, axis=1)
        return y

class MultiHeadSpModel(nn.Module):

    def __init__(
            self, out_dims_head: tp.List[int] = [3, 4, 3, 1], pretrained=False
    ):
        """"""
        self.base_name = "resnet200d"
        self.n_heads = len(out_dims_head)
        super(MultiHeadSpModel, self).__init__()

        # # load base model
        base_model = timm.create_model(
            self.base_name, num_classes=sum(out_dims_head), pretrained=False)

        in_features = base_model.num_features

        if pretrained:
            pretrained_model_path = '../../sp/resnet200d_320_chestx.pth'
            state_dict = dict()
            for k, v in torch.load(pretrained_model_path, map_location='cpu')["model"].items():
                if k[:6] == "model.":
                    k = k.replace("model.", "")
                state_dict[k] = v
            base_model.load_state_dict(state_dict)

        # # remove global pooling and head classifier
        base_model.reset_classifier(0, '')

        # # Shared CNN Bacbone
        self.backbone = base_model

        # # Multi Heads.
        for i, out_dim in enumerate(out_dims_head):
            layer_name = f"head_{i}"
            layer = nn.Sequential(
                SpatialAttentionBlock(in_features, [64, 32, 16, 1]),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, in_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(in_features, out_dim))
            setattr(self, layer_name, layer)

    def forward(self, x):
        """"""
        h = self.backbone(x)
        hs = [
            getattr(self, f"head_{i}")(h) for i in range(self.n_heads)]
        y = torch.cat(hs, axis=1)
        return y

def train_func(dataset_train, model, criterion, optimizer, epoch, n_epochs, batch_size, num_workers, local_rank, model_dir, kernel_type, fold_id):
    model.train()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_sampler.set_epoch(epoch)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    bar = train_loader
    scaler = torch.cuda.amp.GradScaler()
    losses = []
    for batch_idx, (images, targets) in enumerate(bar):
        images, targets = images.cuda(local_rank, non_blocking=True), targets.cuda(local_rank, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss = reduce_mean(loss, torch.distributed.get_world_size())
        losses.append(loss.item())
        smooth_loss = np.mean(losses[-30:])

        if ((batch_idx + 1) % 50 == 0 or batch_idx == len(train_loader) - 1 or batch_idx == 0) and local_rank == 0:
            print('Epoch: {}/{} Step: {}/{} Loss: {:.4f} Aug_loss: {:.4f} Smooth_loss: {:.4f}'.format(epoch, n_epochs, batch_idx + 1, len(train_loader), loss.item(), np.mean(losses), smooth_loss))
            with open(f'{model_dir}{kernel_type}_fold{fold_id}_mh_train.txt', 'a') as txt:
                print('Epoch: {}/{} Step: {}/{} Loss: {:.4f} Aug_loss: {:.4f} Smooth_loss: {:.4f}'.format(epoch, n_epochs, batch_idx + 1, len(train_loader), loss.item(), np.mean(losses), smooth_loss), file=txt)
    loss_train = np.mean(losses)
    return loss_train


def valid_func(dataset_valid, target_cols, model, criterion, valid_batch_size, num_workers):
    model.eval()
    test_sampler = OrderedDistributedSampler(dataset_valid)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=valid_batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True, sampler = test_sampler, drop_last=True)

    bar = valid_loader

    TARGETS = []
    losses = []
    PREDS = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(bar):
            images, targets = images.cuda(), targets.cuda()
            logits = model(images)
            PREDS += [logits.sigmoid()]
            TARGETS += [targets.detach().cpu()]
            loss = criterion(logits, targets)
            loss = reduce_mean(loss, torch.distributed.get_world_size())
            losses.append(loss.item())
    PREDS = gather_tensor(torch.cat(PREDS).cuda())
    TARGETS = gather_tensor(torch.cat(TARGETS).cuda())
    PREDS = PREDS.cpu().numpy()
    TARGETS = TARGETS.cpu().numpy()
    roc_auc = macro_multilabel_auc(TARGETS, PREDS, target_cols)
    roc_auc = np.mean(roc_auc)
    loss_valid = np.mean(losses)
    return loss_valid, roc_auc

def main():

    # Train Parameters
    fold_id = 4

    model_name = 'resnet200d'
    image_size = 640
    seed = 42
    warmup_epo = 2
    init_lr = 4e-4
    batch_size = 32
    valid_batch_size = 64
    n_epochs = 25
    warmup_factor = 10
    num_workers = 4

    debug = False # change this to run on full data
    early_stop = 5
    
    data_dir = './data/ranzcr-clip-catheter-line-classification/'
    model_dir = './model/'
    
    kernel_type = '{}_b{}_lr{}'.format(model_name, batch_size, init_lr)
    if debug:
        kernel_type = 'test'
    

    # Initialize DistributedDataParallel
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    torch.distributed.init_process_group('nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)

    seed_everything(seed)

    # Train/Valid Transformation
    transforms_train = albumentations.Compose([
        albumentations.RandomResizedCrop(image_size, image_size, scale=(0.9, 1), p=1),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(p=0.5),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
        albumentations.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
        albumentations.CLAHE(clip_limit=(1,4), p=0.5),
        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.2),
        albumentations.OneOf([
            albumentations.GaussNoise(var_limit=[10, 50]),
            albumentations.GaussianBlur(),
            albumentations.MotionBlur(),
            albumentations.MedianBlur(),
        ], p=0.2),
        albumentations.Resize(image_size, image_size),
        albumentations.OneOf([
            albumentations.JpegCompression(),
            albumentations.Downscale(scale_min=0.1, scale_max=0.15),
        ], p=0.2),
        albumentations.IAAPiecewiseAffine(p=0.2),
        albumentations.IAASharpen(p=0.2),
        albumentations.Cutout(max_h_size=int(image_size * 0.1), max_w_size=int(image_size * 0.1), num_holes=5, p=0.5),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    ])

    transforms_valid = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    # Prepare Dataset
    df_train = pd.read_csv(data_dir + '/train_folds.csv')
    df_train['file_path'] = df_train.StudyInstanceUID.apply(lambda x: os.path.join(data_dir + '/train/', f'{x}.jpg'))
    if debug:
        df_train = df_train.sample(frac=0.1)
    target_cols = df_train.iloc[:, 1:12].columns.tolist()

    # Load Model
    if debug:
        model = timm.create_model('resnet18', pretrained=False, num_classes=11)
    else:
        # model = MultiHeadModel(model_name, pretrained=True)
        model = MultiHeadSpModel([3, 4, 3, 1], True)

    # model = model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=args.local_rank)

    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr/warmup_factor)

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-7)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=n_epochs, T_mult=1, eta_min=1e-7)

    df_train_this = df_train[df_train['fold'] != fold_id]
    df_valid_this = df_train[df_train['fold'] == fold_id]

    dataset_train = RANZCRDataset(df_train_this, 'train', target_cols, transform=transforms_train)
    dataset_valid = RANZCRDataset(df_valid_this, 'valid', target_cols, transform=transforms_valid)

    log = {}
    roc_auc_max = 0.
    loss_min = 99999
    not_improving = 0

    for epoch in range(1, n_epochs + 1):
    
        # scheduler_warmup.step(epoch - 1)
        lr_scheduler.step(epoch - 1)
        loss_train = train_func(dataset_train, model, criterion, optimizer,
                                epoch, n_epochs, int(batch_size/torch.distributed.get_world_size()), num_workers, args.local_rank,
                                model_dir, kernel_type, fold_id)

        loss_valid, roc_auc = valid_func(dataset_valid, target_cols, model, criterion,
                                         int(valid_batch_size/torch.distributed.get_world_size()), num_workers)

        log['loss_train'] = log.get('loss_train', []) + [loss_train]
        log['loss_valid'] = log.get('loss_valid', []) + [loss_valid]
        log['lr'] = log.get('lr', []) + [optimizer.param_groups[0]["lr"]]
        log['roc_auc'] = log.get('roc_auc', []) + [roc_auc]

        if args.local_rank ==0:
            content = time.ctime() + ' ' + f'Fold {fold_id}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, loss_train: {loss_train:.5f}, loss_valid: {loss_valid:.5f}, roc_auc: {roc_auc:.6f}.'
            print(content)
            with open(f'{model_dir}{kernel_type}_fold{fold_id}_mh.txt', 'a') as txt:
                print(content, file=txt)

            not_improving += 1

            if roc_auc > roc_auc_max:
                print(f'roc_auc_max ({roc_auc_max:.6f} --> {roc_auc:.6f}). Saving model ...')
                torch.save(model.state_dict(), f'{model_dir}{kernel_type}_fold{fold_id}_mh_best_AUC.pth')
                roc_auc_max = roc_auc
                not_improving = 0

            if loss_valid < loss_min:
                loss_min = loss_valid
                torch.save(model.state_dict(), f'{model_dir}{kernel_type}_fold{fold_id}_mh_best_loss.pth')

        if not_improving == early_stop:
            print('Early Stopping...')
            break
        if debug:
            break
    torch.save(model.state_dict(), f'{model_dir}{kernel_type}_fold{fold_id}_mh_final.pth')

if __name__ == '__main__':
    main()