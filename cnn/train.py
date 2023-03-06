import os
import gc
import cv2
import sys
import torch
import random
import argparse
import importlib
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import torch.nn as nn
from shutil import copyfile
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import get_cosine_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
# from torchcontrib.optim import SWA
from warnings import filterwarnings
filterwarnings("ignore")

sys.path.append("models")
sys.path.append("configs")
sys.path.append("dataset")

parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename")
parser.add_argument("-M", "--mode", default='train', help="mode type")
parser.add_argument("-T", "--tta", default=0, help="is use tta for inference")
parser_args = parser.parse_args()

print("[ √ ] Using config file", parser_args.config)
print("[ √ ] Using mode: ", parser_args.mode)

cfg = importlib.import_module(parser_args.config).cfg
NModel = importlib.import_module(cfg.model).NModel
NDataset = importlib.import_module(cfg.dataset).NDataset
torch.multiprocessing.set_sharing_strategy('file_system')

from dataset_3d_3ch_v2 import SimpleClassSampler

cfg.mode = parser_args.mode
cfg.config = parser_args.config

os.makedirs(cfg.out_dir, exist_ok=True)

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def logfile(message):
    print(message)
    with open(log_path, 'a+') as logger:
        logger.write(f'{message}\n')

def get_optimizer(cfg, model):
    params = [
        {
            "params": [param for name, param in model.named_parameters()],
            "lr": cfg.lr,
        }
    ]

    if cfg.optimizer == "Adam":
        optimizer = torch.optim.Adam(params, lr=params[0]["lr"])
    elif cfg.optimizer == "SGD":
        optimizer = torch.optim.SGD(params, lr=params[0]["lr"], momentum=0.9, nesterov=True,)
    elif cfg.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=params[0]["lr"])
        # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, eps=1e-6, betas=(0.9, 0.99))

    return optimizer

def get_scheduler(cfg, optimizer, total_steps=0):
    iter_update = False
    if cfg.scheduler == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40000, gamma=0.8)
    elif cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6, verbose=False)
    elif cfg.scheduler == "linear":
        iter_update = True
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=(total_steps // cfg.batch_size),
        )
        print("num_steps", (total_steps // cfg.batch_size))
    elif cfg.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 25, 30], gamma=0.5, verbose=False)
    elif cfg.scheduler == "cosinewarmup":
        iter_update = True
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup,
            num_training_steps=(total_steps // cfg.batch_size),
        )
    else:
        scheduler = None

    return scheduler, iter_update

def get_dataloader(fold_id = 0):
    train_transform = cfg.train_transform
    val_transform = cfg.val_transform

    df = pd.read_csv(cfg.train_csv_path)
    print(df.shape)

    if cfg.use_oof:
        ffs = []
        for ff in [0,1,2,3,4]:
            if ff != fold_id:
                f_df = pd.read_csv(f'{cfg.pl_path}/oof_f{ff}.csv')
                ffs.append(f_df)
        train_df = pd.concat(ffs)
    else:
        train_df = df[df.fold!=fold_id]

    if not cfg.sampler:
        train_df1 = train_df[train_df.contact==1]
        train_df0 = train_df[train_df.contact==0]

        train_df1 = train_df1.sample(frac=1)
        train_df0 = train_df0.sample(frac=1)
        num_neg = int(cfg.frac*train_df1.shape[0])
        train_df0 = train_df0.head(num_neg)
        train_df = pd.concat([train_df0, train_df1])


    train_df = train_df.sample(frac = 1.0) 
    print(train_df.contact.value_counts())

    train_dataset = NDataset(cfg, train_df, tfms=train_transform, fold_id = fold_id, is_train=True)

    if fold_id < 5:
        val_df = df[df.fold==fold_id]
    else:
        val_df = df[df.fold==0]

    if cfg.mode not in ['val']:
        # val_df = val_df.head(100)
        # val_df = val_df.sample(frac=0.1, random_state=42)
        val_df1 = val_df[val_df.contact==1]
        val_df0 = val_df[val_df.contact==0]
        val_df1 = val_df1.sample(frac=0.1*cfg.val_frac, random_state=42)
        val_df0 = val_df0.sample(frac=0.04*cfg.val_frac, random_state=42)
        val_df = pd.concat([val_df0, val_df1])
        val_df = val_df.sample(frac = 1.0, random_state=42) 
        print(val_df.contact.value_counts())

    val_dataset = NDataset(cfg, val_df, tfms=val_transform,  fold_id = fold_id, is_train = False)

    if not cfg.sampler:
        train_dataloader = DataLoader(train_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=True)
    else:
        train_dataloader = DataLoader(train_dataset,
            batch_size=cfg.batch_size,
            sampler=SimpleClassSampler(train_df, cfg),
            num_workers=cfg.num_workers,
            drop_last=True
            )

    val_dataloader = DataLoader(val_dataset,
        batch_size=cfg.batch_size,
        # batch_size=1,
        num_workers=cfg.num_workers,
        shuffle=False)

    total_steps = len(train_dataloader)*cfg.batch_size

    # for i, batch in enumerate(train_dataset):
    #     img, lb, f = batch
    #     print(img.shape)
    #     img = img.numpy().transpose(1,2,3,0)*255
    #     # out = img[5,:,:,:3]
    #     # mask0 = np.uint8(255*mask[0].numpy())
    #     # print(mask0.shape)
    #     # mask0 = cv2.cvtColor(mask0,cv2.COLOR_GRAY2RGB)
    #     # out = np.hstack([img, mask0])
    #     for ii in range(img.shape[0]):
    #         im = img[ii]
    #         cv2.imwrite(f'{cfg.out_dir}/s{i}_{ii}.jpg', im)
    #     if i>10:
    #         break
    # exit()

    return train_dataloader, val_dataloader, total_steps, val_df

def valid_func(model, val_loader, tta=int(parser_args.tta)):
    if cfg.loss_fn == 'bce':
        loss_cls_fn = nn.BCEWithLogitsLoss()
    elif cfg.loss_fn == 'focal':
        loss_cls_fn = BCEFocalLoss()
    else:
        loss_cls_fn = nn.CrossEntropyLoss()

    y_preds = []
    y_trues = []
    count = 1
    model.eval()
    with torch.no_grad():
        losses = []
        bar = tqdm(val_loader)
        for batch_idx, batch_data in enumerate(bar):
            if cfg.debug and batch_idx>10:
                break
            images, lb, feat = batch_data
            images = images.float().to(device)
            if cfg.use_meta:
                pred = model(images, feat.to(device))
                logit = pred['out1']
            else:
                pred = model(images)
                logit = pred['out1']
                if tta:
                    pred1 = model(images.flip(-1))
                    logit = 0.5*logit + 0.5*pred1['out1']

            if cfg.loss_fn in ['bce', 'focal']:
                loss = loss_cls_fn(logit, lb.to(device).unsqueeze(-1))
            else:
                loss = loss_cls_fn(logit, lb.to(device).long())
            
            losses.append(loss.item())
            smooth_loss = np.mean(losses[:])

            bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')

            y_trues.append(lb.detach().cpu().numpy())
            if cfg.loss_fn in ['bce', 'focal']:
                out = logit.sigmoid().detach().cpu().numpy()
            else:
                out = logit.softmax(-1).detach().cpu().numpy()[:,1]
                # print(out.shape)
            y_preds.append(out)

    y_preds = np.concatenate(y_preds).astype(np.float64)
    y_trues = np.concatenate(y_trues).astype(np.float64)
    print(y_preds.shape, y_trues.shape)

    y_preds = y_preds.reshape(-1)
    y_trues = y_trues.reshape(-1)

    if cfg.mode == 'test':
        acc = 1
        auc = 1
        macro_score = 1
    else:
        acc = matthews_corrcoef(y_trues>0.5, y_preds > 0.5)
        auc = roc_auc_score(y_trues>0.5, y_preds)
        # micro_score = f1_score(y_trues>0.5, y_preds > 0.5, average='micro')
        macro_score = f1_score(y_trues>0.5, y_preds > 0.5, average='macro')

    val_loss = np.mean(losses)
    return val_loss, auc, acc, macro_score, y_preds

def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

if __name__ == '__main__':
    # set_seed(cfg.seed)
    print(cfg.folds)

    if cfg.mode == 'train':
        copyfile(os.path.basename(__file__), os.path.join(cfg.out_dir, os.path.basename(__file__)))
        copyfile(f'configs/{cfg.config}.py', os.path.join(cfg.out_dir, f'{cfg.config}.py'))
        copyfile(f'dataset/dataset_3d_3ch_v2.py', os.path.join(cfg.out_dir, f'dataset_3d_3ch_v2.py'))

    f_preds = []
    models = []
    for fold_id in cfg.folds:
        device = "cuda"
        model = NModel(cfg)

        if cfg.num_freeze > 0 and cfg.mode == 'train':
            for cc, (name, params) in enumerate(model.backbone.named_parameters()):
                # if cc < cfg.num_freeze or 'layer1.' in name or 'layer2.' in name or 'layer3.' in name:
                if cc < cfg.num_freeze or 'layer1.' in name:
                # if cc < cfg.num_freeze:
                    print(f'layer {cc} {name} is frozen!')
                    params.requires_grad = False

        model.to(device)

        if cfg.mode == 'train':
            log_path = f'{cfg.out_dir}/log_f{fold_id}.txt'
            logfile(f'====== FOLD {fold_id} =======')

            if len(cfg.load_weight) > 10:
                if '.pth' in cfg.load_weight:
                    load_weight = cfg.load_weight
                else:
                    load_weight = f'{cfg.load_weight}_last_f{(fold_id)%5}.pth'

                logfile(f'load pretrained weight {load_weight}!!!')
                state_dict = torch.load(load_weight, map_location=device)  # load checkpoint
                if 'state_dict' in state_dict.keys():
                    state_dict = state_dict['state_dict']
                state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=[])  # intersect
                model.load_state_dict(state_dict, strict=False)  # load
                logfile('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), load_weight))  # report
                del state_dict
                gc.collect()
            
            train_loader, valid_loader, total_steps, val_df =  get_dataloader(fold_id)
            total_steps = total_steps*cfg.epochs

            optimizer = get_optimizer(cfg, model)

            if cfg.use_swa:
                optimizer = SWA(
                    optimizer,
                    swa_start=500,
                    swa_freq=50,
                    swa_lr=None)

            scheduler, iter_update = get_scheduler(cfg, optimizer, total_steps)

            best_loss = 1e6
            best_auc = 0
            best_macro = 0
            step = 0
            for epoch in range(1,cfg.epochs+1):
                if not iter_update:
                    scheduler.step(epoch)
                if cfg.loss_fn == 'bce':
                    loss_cls_fn = nn.BCEWithLogitsLoss()
                elif cfg.loss_fn == 'focal':
                    loss_cls_fn = BCEFocalLoss()
                else:
                    loss_cls_fn = nn.CrossEntropyLoss()

                scaler = torch.cuda.amp.GradScaler(enabled=cfg.apex)

                model.train()
                losses = []
                bar = tqdm(train_loader)
                for batch_idx, batch_data in enumerate(bar):
                    step +=1
                    if cfg.debug and batch_idx>10:
                        break
                    images, lb, feat = batch_data
                    # print(images.shape, lb.shape)

                    if cfg.use_meta:
                        pred = model(images.float().to(device), feat.to(device))
                    else:
                        pred = model(images.float().to(device))
                    logit = pred['out1']
                    # print(logit.shape)
                    if cfg.loss_fn in ['bce', 'focal']:
                        loss = loss_cls_fn(logit, lb.to(device).unsqueeze(-1))
                    else:
                        loss = loss_cls_fn(logit, lb.to(device).long())

                    if cfg.use_oof:
                        loss1 = loss_cls_fn(logit, feat.to(device).unsqueeze(-1))
                        loss = 0.5*loss + 0.5*loss1
                    
                    loss.backward()

                    scaler.step(optimizer)
                    scaler.update()

                    if batch_idx%100 == 0 and batch_idx>300 and cfg.use_swa:
                        optimizer.update_swa()

                    optimizer.zero_grad()

                    if iter_update:
                        scheduler.step()

                    losses.append(loss.item())
                    smooth_loss = np.mean(losses[-2000:])

                    bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}, LR {scheduler.get_lr()[0]:.6f}')

                train_loss = np.mean(losses)

                loss_valid, auc, acc, macro_score, y_preds = valid_func(model, valid_loader)
                metric = auc - loss_valid
                logfile(f'[EPOCH] {epoch}, train_loss: {train_loss:.6f},  Val loss: {loss_valid:.6f}, AUC: {auc:.5f} ACC {acc:.5f} macro_score {macro_score:.5f}')

                if best_macro <= metric:
                    logfile(f'[EPOCH] {epoch} ===============> best_macro ({best_macro:.6f} --> {metric:.6f}). Saving model .......!!!!\n')
                    torch.save(model.state_dict(), f'{cfg.out_dir}/best_metric_f{fold_id}.pth')
                    best_macro = metric

            torch.save(model.state_dict(), f'{cfg.out_dir}/{cfg.config}_last_f{fold_id}.pth')

            del model, scheduler, optimizer
            gc.collect()
        elif cfg.mode == 'val':
            if fold_id > 4:
                continue
            # chpt_path = f'{cfg.out_dir}/best_metric_f{fold_id}.pth'
            chpt_path = f'{cfg.out_dir}/{cfg.config}_last_f{fold_id}.pth'

            print(f' load {chpt_path}!')
            checkpoint = torch.load(chpt_path, map_location="cpu")
            model.load_state_dict(checkpoint)

            train_loader, valid_loader, total_steps, val_df =  get_dataloader(fold_id)
            loss_valid, auc, acc, macro_score, y_preds = valid_func(model, valid_loader)
            print(f'Val loss: {loss_valid:.6f} AUC: {auc:.5f} ACC {acc:.5f} macro_score {macro_score:.5f}')
            val_df['pred'] = y_preds
            f_preds.append(val_df)

            if int(parser_args.tta) == 1:
                val_df.to_csv(f'{cfg.out_dir}/oof_f{fold_id}_tta.csv', index=False)
            else:
                val_df.to_csv(f'{cfg.out_dir}/oof_f{fold_id}.csv', index=False)

            del model, checkpoint
            gc.collect()

    if cfg.mode == 'val':
        oof_df = pd.concat(f_preds)
        if int(parser_args.tta) == 1:
            oof_df.to_csv(f'{cfg.out_dir}/oof_tta.csv', index=False)
        else:
            oof_df.to_csv(f'{cfg.out_dir}/oof.csv', index=False)
