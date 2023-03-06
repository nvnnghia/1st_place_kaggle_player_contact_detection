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
    NAME = "pre_p_xgb"

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
        # print(pred_i[:10], y_valid[:10])

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
    oof_df.to_csv(f'{cfg.EXP_MODEL}/xgb_oof_pair_fe.csv', index=False)

    gt = oof_df.contact.values
    all_pos = np.sum(gt==1)
    for thres in range(20,60):
        thres = thres*0.01
        pred = 1*(oof_df.pred.values > thres)
        tp = np.sum((gt==1)*(pred==1))
        pred_pos = np.sum(pred==1)

        score = matthews_corrcoef(gt, pred > thres)

        print(f'thres {thres:.4f} tp {tp} all_pos {all_pos:.4f} pred_pos {pred_pos:.4f}, score {score:.4f}')


def extract_feat(idx, trk_dict, step, nan_val=0, window_size=10):
    pos_code = {'CB':1, 'DE':2, 'FS':3, 'TE':4, 'ILB':5, 'OLB':6, 'T':7, 'G':8, 'C':9, 'QB':10, 'WR':11, 'RB':12, 'NT':13, 'DT':14,
        'MLB':15, 'SS':16, 'OT':17, 'LB':18, 'OG':19, 'SAF':20, 'DB':21, 'LS':22, 'K':23, 'P':24, 'FB':25, 'S':26, 'DL':27, 'HB':28}

    if idx not in trk_dict:
        item = {'s': nan_val, 'dis': nan_val, 'dir': nan_val, 'o': nan_val, 'a': nan_val, 'sa': nan_val, 'x': nan_val, 'y': nan_val, 't': nan_val}
        # item[f'pos'] = nan_val
    else:
        if step in trk_dict[idx]:
            item = {'s': trk_dict[idx][step]['s'], 'dis': trk_dict[idx][step]['dis'], 'dir': trk_dict[idx][step]['dir'], 'o': trk_dict[idx][step]['o']} 
            item['a'] = trk_dict[idx][step]['a']
            item['sa'] = trk_dict[idx][step]['sa']
            item['x'] = trk_dict[idx][step]['x']
            item['y'] = trk_dict[idx][step]['y'] 
            item['t'] = trk_dict[idx][step]['t'] 
        else:
            item = {'s': nan_val, 'dis': nan_val, 'dir': nan_val, 'o': nan_val, 'a': nan_val, 'sa': nan_val, 'x': nan_val, 'y': nan_val, 't': nan_val}

    return item


def calc_dist(idx1, idx2, trk_dict, step, nan_val=0):
    if idx1 not in trk_dict or idx2 not in trk_dict:
        return nan_val, nan_val, nan_val
    else:
        if step not in trk_dict[idx1] or step not in trk_dict[idx2]:
            return nan_val, nan_val, nan_val

        x1 = trk_dict[idx1][step]['x']
        y1 = trk_dict[idx1][step]['y'] 

        x2 = trk_dict[idx2][step]['x']
        y2 = trk_dict[idx2][step]['y'] 

        dist = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

        sa_dif = trk_dict[idx1][step]['sa'] - trk_dict[idx2][step]['sa'] #a
        a_dif = trk_dict[idx1][step]['a'] - trk_dict[idx2][step]['a'] #a

        return dist, a_dif, sa_dif

def feature_engineering():
    train_df = pd.read_csv('../data/train_folds.csv')
    train_df = train_df[train_df.nfl_player_id_2 != 'G']
    train_df = train_df[train_df.distance<4.0]
    train_df['step'] = train_df['contact_id'].apply(lambda x: int(x.split('_')[2]))
    train_df['vid'] = train_df['contact_id'].apply(lambda x: '_'.join(x.split('_')[:2]))

    print(train_df.shape)

    train_df['game_play'] = train_df['vid']

    trk_dict = np.load('../data/trk_dict.npy', allow_pickle=True).item()

    det_dict = np.load('../data/det_dict.npy', allow_pickle=True).item()

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

        item1 = extract_feat(idx, trk_dict, step, nan_val=nan_val, window_size=window_size)
        item2 = extract_feat(idx2, trk_dict, step, nan_val=nan_val, window_size=window_size)

        item = {}
        for k, val in item1.items():
            if k in ['t']:
                if val == item2[k]:
                    item[k] = 0
                else:
                    item[k] = 1
                continue

            item[k] = val 
            item[f'{k}_2'] = item2[k]

            if k not in ['pos']:
                item[f'{k}_dif'] = val - item2[k]

            if k in ['o', 'dir']:
                item[f'{k}_s'] = math.sin(val)
                item[f'{k}_c'] = math.cos(val)
                item[f'{k}_s2'] = math.sin(item2[k])
                item[f'{k}_c2'] = math.cos(item2[k])
                item[f'{k}_sd'] = math.sin(val - item2[k])
                item[f'{k}_cd'] = math.cos(val - item2[k])

            if k in ['o', 'dir']:
                item[f'{k}_s'] = math.sin(math.pi*val/180)
                item[f'{k}_c'] = math.cos(math.pi*val/180)
                item[f'{k}_s2'] = math.sin(math.pi*item2[k]/180)
                item[f'{k}_c2'] = math.cos(math.pi*item2[k]/180)
                item[f'{k}_sd'] = math.sin(math.pi*(val - item2[k])/180)
                item[f'{k}_cd'] = math.cos(math.pi*(val - item2[k])/180)


        item['distance'] = row['distance']
        item['step'] = row['step']

        for i in range(20):
            dist, a_dif, sa_dif = calc_dist(idx, idx2, trk_dict, step+1+i, nan_val=np.nan)
            item[f'dist_{i}'] = dist
            if i<20:
                item[f'a_{i}'] = a_dif
                item[f'sa_{i}'] = sa_dif

        for i in range(20):
            dist, a_dif, sa_dif = calc_dist(idx, idx2, trk_dict, step-1-i, nan_val=np.nan)
            item[f'dist_p{i}'] = dist
            if i<20:
                item[f'a_p{i}'] = a_dif
                item[f'sa_p{i}'] = sa_dif

        idx1 = int(row['nfl_player_id_1'])
        idx2 = int(row['nfl_player_id_2'])
        step = row['step']
        frame = int(row['frame'])+6

        for view in ['Sideline', 'Endzone']:
            v_vid = vid + '_' + view
            area1_list = []
            area2_list = []
            for ff in range(-30,30,2):
                fr = frame + ff
                if fr in det_dict[v_vid] and idx1 in det_dict[v_vid][fr] and idx2 in det_dict[v_vid][fr]: 
                    x1, y1, w1, h1 = det_dict[v_vid][fr][idx1]['box']
                    x2, y2, w2, h2 = det_dict[v_vid][fr][idx2]['box']

                    x1 = x1 + w1/2
                    y1 = y1 + h1/2
                    x2 = x2 + w2/2
                    y2 = y2 + h2/2
                    dist = math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))

                    item[f'{view}_{ff}_dist'] = dist
                    area1_list.append(w1*h1)
                    area2_list.append(w2*h2)
                else:
                    item[f'{view}_{ff}_dist'] = np.nan

            if len(area2_list)>0:
                item[f'{view}_area1'] = np.mean(area1_list)
                item[f'{view}_area2'] = np.mean(area2_list)
            else:
                item[f'{view}_area1'] = np.nan
                item[f'{view}_area2'] = np.nan

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

train_df, feature_cols = feature_engineering()
gc.collect()

cfg = setup(Config)

oof_pred = fit_xgboost(cfg, cfg.xgb_params, add_suffix="_xgb_1st")
