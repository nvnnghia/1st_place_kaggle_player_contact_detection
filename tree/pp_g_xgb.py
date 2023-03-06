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

from shutil import copyfile

class Config:
    NAME = "pp_g_xgb"

    seed = 42
    num_fold = 5
    
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate':0.005,
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
            early_stopping_rounds=200,
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

        for thres in [0.05,0.1,0.2,0.3, 0.4, 0.5]:
            pred = 1*(pred_i > thres)
            tp = np.sum((gt==1)*(pred==1))
            pred_pos = np.sum(pred==1)

            score = matthews_corrcoef(gt, pred > thres)

            print(f'thres {thres:.4f} tp {tp} all_pos {all_pos:.4f} pred_pos {pred_pos:.4f}, score {score:.4f}')

        score = round(roc_auc_score(y_valid, pred_i), 5)
        print(f'Performance of the prediction: {score}\n')
        del model; gc.collect()

    oof_df = pd.concat(oof_pred)
    oof_df.to_csv(f'{cfg.EXP_MODEL}/xgb_G_oof{add_suffix}.csv', index=False)

    gt = oof_df.contact.values
    all_pos = np.sum(gt==1)
    for thres in range(20,50):
        thres = thres*0.01
        pred = 1*(oof_df.pred.values > thres)
        tp = np.sum((gt==1)*(pred==1))
        pred_pos = np.sum(pred==1)

        score = matthews_corrcoef(gt, pred > thres)

        print(f'thres {thres:.4f} tp {tp} all_pos {all_pos:.4f} pred_pos {pred_pos:.4f}, score {score:.4f}')
    return oof_df 

def get_oof(config_name):
    dfs = []
    for i in [0,1,2,3,4]:
        df_pred = pd.read_csv(f'../cnn/outputs/{config_name}/oof_f{i}.csv')
        dfs.append(df_pred)
    df_pred = pd.concat(dfs).reset_index(drop=True)

    pred_step_dict = {}
    for i, row in df_pred.iterrows():
        idx = row['path'].split('/')[-1]
        step = int(idx.split('_')[-1])
        root = '_'.join(idx.split('_')[:-2])
        idx = f'{root}_{step}'
        pred_step_dict[idx] = row['pred']

    return pred_step_dict

def feature_engineering():
    train_df = pd.read_csv('pre_g_xgb/model/xgb_G_oof.csv')

    train_df['step'] = train_df['contact_id'].apply(lambda x: int(x.split('_')[2]))
    train_df['vid'] = train_df['contact_id'].apply(lambda x: '_'.join(x.split('_')[:2]))
    train_df['nfl_player_id_1'] = train_df['contact_id'].apply(lambda x: int(x.split('_')[3]))
    train_df['nfl_player_id_2'] = 'G'

    trk_dict = np.load('../data/trk_dict.npy', allow_pickle=True).item()

    pred_step_dict = get_oof('r50ir_csn_c11_m1_d2_G')
    pred_step_dict2 = get_oof('r50ir_csn_c15_m1_d2_G')

    pred_step_dict1 = {}
    for i, row in train_df.iterrows():
        c_id = row['contact_id']
        fr_id = int(row['frame'])
        pred = row['pred']
        gk, gp, st, idx1, idx2 = c_id.split('_')
        idx = f'{gk}_{gp}_{idx1}_{idx2}_{st}'
        pred_step_dict1[idx] = row['pred']

    results = []
    for i, row in tqdm(train_df.iterrows()):
        idx1 = row['nfl_player_id_1']
        idx2 = row['nfl_player_id_2']
        fr_id = int(row['frame'])
        vid = row['vid']
        step = row['step']
        idx = f'{vid}_{idx1}_{idx2}_{fr_id:04d}'
        idx = f'{idx}_{step}'

        contact = row['contact']

        item = {'contact_id':row['contact_id'], 'contact':contact, 'step':row['step'], 'frame': row['frame'], 'fold':row['fold']}
        item['nfl_player_id_1'] = row['nfl_player_id_1']
        item['nfl_player_id_2'] = row['nfl_player_id_2']
        item['prob1'] = row['pred']
        item['vid'] = row['vid']

        for i in range(-15,15):
            this_idx = f'{vid}_{idx1}_{idx2}_{step+i}'

            prob = 0
            weight = 0
            if this_idx in pred_step_dict:
                prob += pred_step_dict[this_idx]
                weight += 1

            if this_idx in pred_step_dict2:
                prob += pred_step_dict2[this_idx]
                weight += 1

            if weight > 0:
                item[f'prob_{i}'] = prob/weight
            else:
                item[f'prob_{i}'] = np.nan

            if this_idx in pred_step_dict1:
                item[f'prob1_{i}'] = pred_step_dict1[this_idx]
            else:
                item[f'prob1_{i}'] = np.nan

        results.append(item) 

    train_df = pd.DataFrame(results)

    results = []
    for i, row in tqdm(train_df.iterrows()):
        vid = row['vid']
        idx = row['nfl_player_id_1']
        idx = f'{vid}_{idx}'
        step = row['step']

        item = {}

        for i in range(-15,15):
            if i>-10 and i < 10:
                item[f'prob_{i}'] = row[f'prob_{i}']
                item[f'prob1_{i}'] = row[f'prob1_{i}']
            item[f'prob3_{i}'] = 0.85*row[f'prob_{i}']  + 0.15*row[f'prob1_{i}']

        feature_cols = list(item.keys())

        item['fold'] = row['fold']
        item['contact'] = row['contact']
        item['contact_id'] = row['contact_id']
        item['frame'] = row['frame']


        results.append(item)


    train_df = pd.DataFrame(results)

    return train_df, feature_cols

cfg = setup(Config)
copyfile(os.path.basename(__file__), os.path.join(cfg.EXP_MODEL, os.path.basename(__file__)))

train_df, feature_cols = feature_engineering()
gc.collect()

oof_pred = fit_xgboost(cfg, cfg.xgb_params, add_suffix="_xgb_1st")
