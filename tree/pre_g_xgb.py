# modify from https://www.kaggle.com/code/columbia2131/nfl-player-contact-detection-simple-xgb-baseline
import os
import gc
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import math
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
)
import xgboost as xgb
import torch
import scipy.stats as sss

class Config:
    NAME = "pre_g_xgb"

    seed = 42
    num_fold = 5
    
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate':0.03,
        'tree_method':'hist' if not torch.cuda.is_available() else 'gpu_hist'
    }

def setup(cfg):
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # set dirs
    cfg.EXP = cfg.NAME

    cfg.EXP_MODEL = os.path.join(cfg.EXP, 'model')
    cfg.EXP_FIG = os.path.join(cfg.EXP, 'fig')
    cfg.EXP_PREDS = os.path.join(cfg.EXP, 'preds')

    # make dirs
    for d in [cfg.EXP_MODEL, cfg.EXP_FIG, cfg.EXP_PREDS]:
        os.makedirs(d, exist_ok=True)
        
    return cfg


# xgboost code
def fit_xgboost(cfg, params, add_suffix=''):
    oof_pred = []
    for fold in [0,1,2,3,4]:
        if fold == -1: continue

        x_train = train_df[train_df.fold!=fold][feature_cols]
        y_train = train_df[train_df.fold!=fold]['contact']

        x_val = train_df[train_df.fold==fold]

        x_valid = x_val[feature_cols]

        y_valid = train_df[train_df.fold==fold]['contact']

        print(x_train.shape, x_valid.shape)

        xgb_train = xgb.DMatrix(x_train, label=y_train)
        xgb_valid = xgb.DMatrix(x_valid, label=y_valid)
        evals = [(xgb_train,'train'),(xgb_valid,'eval')]

        model = xgb.train(
            params,
            xgb_train,
            num_boost_round=10_000,
            early_stopping_rounds=300,
            evals=evals,
            verbose_eval=100,
        )

        model_path = os.path.join(cfg.EXP_MODEL, f'xgb_fold{fold}{add_suffix}.model')
        model.save_model(model_path)
        model = xgb.Booster()
        model.load_model(model_path)

        dvalid = xgb.DMatrix(x_valid)

        pred_i = model.predict(dvalid) 
        print(pred_i.shape)
        # print(pred_i[:10], y_valid[:10])

        x_val['pred'] = pred_i
        x_val = x_val[['contact_id', 'fold', 'contact', 'pred', 'frame']]
        oof_pred.append(x_val)

        gt = y_valid.values
        all_pos = np.sum(gt==1)

        for thres in [0.0002,0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05,0.1,0.2,0.3, 0.4, 0.5]:
            pred = 1*(pred_i > thres)
            tp = np.sum((gt==1)*(pred==1))
            pred_pos = np.sum(pred==1)

            score = matthews_corrcoef(gt, pred > thres)

            print(f'thres {thres:.4f} tp {tp} all_pos {all_pos:.4f} pred_pos {pred_pos:.4f}, score {score:.4f}')

        score = round(roc_auc_score(y_valid, pred_i), 5)
        print(f'Performance of the prediction: {score}\n')
        del model; gc.collect()

    oof_df = pd.concat(oof_pred)
    oof_df.to_csv(f'{cfg.EXP_MODEL}/xgb_G_oof.csv', index=False)

    gt = oof_df.contact.values
    all_pos = np.sum(gt==1)
    for thres in [0.001, 0.002, 0.01, 0.02, 0.03, 0.04, 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7]:
        pred = 1*(oof_df.pred.values > thres)
        tp = np.sum((gt==1)*(pred==1))
        pred_pos = np.sum(pred==1)

        score = matthews_corrcoef(gt, pred > thres)

        print(f'thres {thres:.4f} tp {tp} all_pos {all_pos:.4f} pred_pos {pred_pos:.4f}, score {score:.4f}')


def feature_engineering():
    train_df = pd.read_csv('../data/train_folds.csv')
    train_df = train_df[train_df.nfl_player_id_2 == 'G']

    train_df['step'] = train_df['contact_id'].apply(lambda x: int(x.split('_')[2]))
    train_df['vid'] = train_df['contact_id'].apply(lambda x: '_'.join(x.split('_')[:2]))
    train_df['nfl_player_id_1'] = train_df['contact_id'].apply(lambda x: int(x.split('_')[3]))
    train_df['nfl_player_id_2'] = 'G'

    trk_dict = np.load('../data/trk_dict.npy', allow_pickle=True).item()

    results = []
    nan_val = 0
    window_size = 25
    for i, row in tqdm(train_df.iterrows()):
        vid = row['vid']
        idx = row['nfl_player_id_1']
        idx = f'{vid}_{idx}'
        step = row['step']

        agg_dict = {'s': [], 'dis': [], 'dir': [], 'o': [], 'a': [], 'sa': [], 'x': [], 'y': []}

        if idx not in trk_dict:
            item = {'s': nan_val, 'dis': nan_val, 'dir': nan_val, 'o': nan_val, 'a': nan_val, 'sa': nan_val, 'x': nan_val, 'y': nan_val}
            for i in range(-window_size,window_size):
                item[f's_{i}'] = nan_val
                item[f'dis_{i}'] = nan_val
                item[f'dir_{i}'] = nan_val
                item[f'o_{i}'] = nan_val
                item[f'a_{i}'] = nan_val
                item[f'sa_{i}'] = nan_val
                item[f'x_{i}'] = nan_val
                item[f'y_{i}'] = nan_val
        else:
            if step in trk_dict[idx]:
                item = {'s': trk_dict[idx][step]['s'], 'dis': trk_dict[idx][step]['dis'], 'dir': trk_dict[idx][step]['dir'], 'o': trk_dict[idx][step]['o']} 
                item['a'] = trk_dict[idx][step]['a']
                item['sa'] = trk_dict[idx][step]['sa']
                item['x'] = trk_dict[idx][step]['x']
                item['y'] = trk_dict[idx][step]['y'] 
            else:
                item = {'s': nan_val, 'dis': nan_val, 'dir': nan_val, 'o': nan_val, 'a': nan_val, 'sa': nan_val, 'x': nan_val, 'y': nan_val}
            for i in range(-window_size,window_size):
                step1 = step + i 

                if i == 0:
                    continue
                
                if step1 in trk_dict[idx]:
                    item[f's_{i}'] = item[f's'] - trk_dict[idx][step1]['s']
                    item[f'dis_{i}'] = item[f'dis'] - trk_dict[idx][step1]['dis']
                    item[f'dir_{i}'] = item[f'dir'] - trk_dict[idx][step1]['dir']
                    item[f'o_{i}'] = item[f'o'] - trk_dict[idx][step1]['o']
                    item[f'a_{i}'] = item[f'a'] - trk_dict[idx][step1]['a']
                    item[f'sa_{i}'] = item[f'sa'] - trk_dict[idx][step1]['sa']
                    item[f'x_{i}'] = item[f'x'] - trk_dict[idx][step1]['x']
                    item[f'y_{i}'] = item[f'y'] - trk_dict[idx][step1]['y']
                else:
                    item[f's_{i}'] = nan_val
                    item[f'dis_{i}'] = nan_val
                    item[f'dir_{i}'] = nan_val
                    item[f'o_{i}'] = nan_val
                    item[f'a_{i}'] = nan_val
                    item[f'sa_{i}'] = nan_val
                    item[f'x_{i}'] = nan_val
                    item[f'y_{i}'] = nan_val


        item['step'] = row['step']
        feature_cols = list(item.keys())

        item['fold'] = row['fold']
        item['contact'] = row['contact']
        item['contact_id'] = row['contact_id']
        item['frame'] = row['frame']

        results.append(item)


    train_df = pd.DataFrame(results)

    return train_df, feature_cols

train_df, feature_cols = feature_engineering()
gc.collect()

cfg = setup(Config)

oof_pred = fit_xgboost(cfg, cfg.xgb_params, add_suffix="_xgb_1st")
