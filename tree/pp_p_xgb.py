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
import joblib 

class Config:
    NAME = "pp_p_xgb"

    seed = 42
    num_fold = 5
    
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate':0.01,
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

        x_val['pred'] = pred_i
        x_val = x_val[['contact_id', 'fold', 'contact', 'pred', 'frame', 'nfl_player_id_1', 'nfl_player_id_2']]
        oof_pred.append(x_val)

        gt = y_valid.values
        all_pos = np.sum(gt==1)

        for thres in [0.002, 0.01, 0.02, 0.03, 0.04, 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7]:
            pred = 1*(pred_i > thres)
            tp = np.sum((gt==1)*(pred==1))
            pred_pos = np.sum(pred==1)

            score = matthews_corrcoef(gt, pred > thres)

            print(f'thres {thres:.4f} tp {tp} all_pos {all_pos:.4f} pred_pos {pred_pos:.4f}, score {score:.4f}')

        score = round(roc_auc_score(y_valid, pred_i), 5)
        print(f'Performance of the prediction: {score}\n')
        del model; gc.collect()

    oof_df = pd.concat(oof_pred)
    oof_df.to_csv(f'{cfg.EXP_MODEL}/xgb_oof_pair_en.csv', index=False)

    gt = oof_df.contact.values
    all_pos = np.sum(gt==1)
    for thres in range(20,50):
        thres = thres*0.01
        pred = 1*(oof_df.pred.values > thres)
        tp = np.sum((gt==1)*(pred==1))
        pred_pos = np.sum(pred==1)

        score = matthews_corrcoef(gt, pred > thres)

        print(f'thres {thres:.4f} tp {tp} all_pos {all_pos:.4f} pred_pos {pred_pos:.4f}, score {score:.4f}')



def get_oof(cfg_name, is_skip=False, suf=''):
    dfs = []
    for i in [0,1,2,3,4]:
        df_pred2 = pd.read_csv(f'../cnn/outputs/{cfg_name}/oof_f{i}_s6{suf}.csv')
        dfs.append(df_pred2)
    df_pred2 = pd.concat(dfs).reset_index(drop=True)

    pred_step_dict2 = {}
    for i, row in df_pred2.iterrows():
        contact_id = row['contact_id']
        gk, gp, step, idx1, idx2 = contact_id.split('_')
        idx = f'{gk}_{gp}_{idx1}_{idx2}_{step}'
        pred = row['pred']
        # if pred < 0.02 and is_skip:
        #     pred = 0
        pred_step_dict2[idx] = pred

    return pred_step_dict2

def get_meta(path):
    xgb_df = pd.read_csv(path)
    xgb_dict = {}
    for i, row in xgb_df.iterrows():
        c_id = row['contact_id']
        fr_id = int(row['frame'])
        pred = row['pred']
        gk, gp, st, idx1, idx2 = c_id.split('_')
        idx = f'{gk}_{gp}_{idx1}_{idx2}_{st}'
        xgb_dict[idx] = pred

    return xgb_dict

def feature_engineering():
    train_df = pd.read_csv('pre_p_xgb/model/xgb_oof_pair_fe.csv')

    train_df['step'] = train_df['contact_id'].apply(lambda x: int(x.split('_')[2]))
    train_df['vid'] = train_df['contact_id'].apply(lambda x: '_'.join(x.split('_')[:2]))
    print(train_df.shape)

    pred_step_dict1 = get_oof('r50ir_csn_c15_m1_d2', True, suf='_xgb')

    xgb_dict = get_meta('pre_p_xgb/model/xgb_oof_pair_fe.csv')

    results = []
    for i, row in tqdm(train_df.iterrows()):
        idx1 = row['nfl_player_id_1']
        idx2 = row['nfl_player_id_2']
        fr_id = int(row['frame'])
        vid = row['vid']
        idx = f'{vid}_{idx1}_{idx2}_{fr_id:04d}'
        step = row['step']
        idx = f'{idx}_{step}'

        item = {'contact_id':row['contact_id'], 'contact':row['contact'], 'step':row['step'], 'frame': row['frame'], 'fold':row['fold']}
        item['nfl_player_id_1'] = row['nfl_player_id_1']
        item['nfl_player_id_2'] = row['nfl_player_id_2']
        item['vid'] = row['vid']

        for i in range(-10,10):
            this_idx = f'{vid}_{idx1}_{idx2}_{step+i}'
            prob = 0
            weight = 0

            if this_idx in pred_step_dict1:
                prob += pred_step_dict1[this_idx]
                weight += 1


            if weight > 0:
                item[f'prob_{i}'] = prob/weight
            else:
                item[f'prob_{i}'] = np.nan

            if this_idx in xgb_dict:
                item[f'prob1_{i}'] = xgb_dict[this_idx]
            else:
                item[f'prob1_{i}'] = np.nan

        results.append(item) 

    train_df = pd.DataFrame(results)

    results = []
    nan_val = np.nan
    window_size = 2
    for i, row in tqdm(train_df.iterrows()):
        vid = row['vid']
        idx = row['nfl_player_id_1']
        idx = f'{vid}_{idx}'

        idx2 = row['nfl_player_id_2']
        idx2 = f'{vid}_{idx2}'

        step = row['step']


        item = {}

        for i in range(-10,10):
            item[f'prob4_{i}'] = 0.8*row[f'prob_{i}'] + 0.2*row[f'prob1_{i}']
        feature_cols = list(item.keys())

        item['fold'] = row['fold']
        item['contact'] = row['contact']
        item['contact_id'] = row['contact_id']
        item['frame'] = row['frame']
        item['nfl_player_id_1'] = row['nfl_player_id_1']
        item['nfl_player_id_2'] = row['nfl_player_id_2']

        results.append(item)

    train_df = pd.DataFrame(results)

    return train_df, feature_cols

cfg = setup(Config)

train_df, feature_cols = feature_engineering()
gc.collect()

oof_pred = fit_xgboost(cfg, cfg.xgb_params, add_suffix="_xgb_1st")
