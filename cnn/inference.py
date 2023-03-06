import numpy as np 
import cv2
import os 
import pandas as pd 
from tqdm import tqdm 
import random
import sys
sys.path.append("models")
import torch
from model_csn1 import NModel
import albumentations as A
import gc 
import time 
import copy 

def get_sample(vid, fr_id, idx1, idx2, window_size = 20, out_size = 128, stride=4):
    ws = []
    hs = []
    for fr in range(fr_id + window_size[0], fr_id + window_size[-1] + 1):
        if fr in det_dict[vid]:
            if idx1 in det_dict[vid][fr] and idx2 in det_dict[vid][fr]: 
                x, y, w, h = det_dict[vid][fr][idx1]['box']
                ws.append(w)
                hs.append(h)
                x, y, w, h = det_dict[vid][fr][idx2]['box']
                ws.append(w)
                hs.append(h)

    if len(ws)>0:
        crop_size = int(5*max(np.mean(ws), np.mean(hs)))
    else:
        crop_size = out_size

    bboxes = []
    for fr in range(fr_id + window_size[0], fr_id + window_size[-1] + 1):
        if fr in det_dict[vid]:
            if idx1 in det_dict[vid][fr] and idx2 in det_dict[vid][fr]: 
                x, y, w, h = det_dict[vid][fr][idx1]['box']
                x1 = x + w/2
                y1 = y + h/2

                x, y, w, h = det_dict[vid][fr][idx2]['box']
                x2 = x + w/2
                y2 = y + h/2

                xc = 0.5*x1 + 0.5*x2
                yc = 0.5*y1 + 0.5*y2

                bboxes.append([xc-crop_size, yc-crop_size, xc+crop_size, yc+crop_size])
            else:
                bboxes.append([np.nan, np.nan, np.nan, np.nan])
        else:
            bboxes.append([np.nan, np.nan, np.nan, np.nan])

    bboxes = pd.DataFrame(bboxes).interpolate(limit_direction='both').values
    images = []
    masks1 = []
    masks2 = []
    empty_count = 0
    for i, ii in enumerate(window_size):
        if bboxes.sum() > 0:
            fr = ii + fr_id
            path = f'{vid}_{fr}'
            
            if path in image_dict:
                image = image_dict[path]
            else:
                image = np.zeros((720, 1280,3), dtype = np.uint8)
                empty_count +=1

            mask1 = np.zeros((720, 1280), dtype = np.uint8)
            mask2 = np.zeros((720, 1280), dtype = np.uint8)

            # x1, y1, x2, y2 = list(map(int, bboxes[i]))
            x1, y1, x2, y2 = list(map(int, bboxes[ii-window_size[0]]))

            y1 = y1 + int(0.2*crop_size)
            x2 = x1 + crop_size*2
            y2 = y1 + crop_size*2

            if fr in det_dict[vid]:
                if idx1 in det_dict[vid][fr]: 
                    x, y, w, h = det_dict[vid][fr][idx1]['box']
                    # mask1[y:y+h, x:x+w] = 255
                    cv2.circle(mask1, (x+w//2, y+h//2), int(0.25*h+0.25*w), 255, thickness=-1)

                if idx2 in det_dict[vid][fr]:
                    x, y, w, h = det_dict[vid][fr][idx2]['box']
                    # mask2[y:y+h, x:x+w] = 255
                    cv2.circle(mask2, (x+w//2, y+h//2), int(0.25*h+0.25*w), 255, thickness=-1)

            crop = image[y1:y2, x1:x2]
            crop_mask1 = mask1[y1:y2, x1:x2]
            crop_mask2 = mask2[y1:y2, x1:x2]

            cr_y, cr_x = crop.shape[:2]
            if cr_x == crop_size*2 and cr_y == crop_size*2:
                crop = cv2.resize(crop, (out_size*2,out_size*2))
                crop_mask1 = cv2.resize(crop_mask1, (out_size*2,out_size*2))
                crop_mask2 = cv2.resize(crop_mask2, (out_size*2,out_size*2))
                images.append(crop)
                masks1.append(crop_mask1)
                masks2.append(crop_mask2)
            else:
                tmp_crop =  np.zeros((crop_size*2, crop_size*2,3), dtype = np.uint8)
                tmp_mask1 =  np.zeros((crop_size*2, crop_size*2), dtype = np.uint8)
                tmp_mask2 =  np.zeros((crop_size*2, crop_size*2), dtype = np.uint8)
                if x1 < 0:
                    if y2>=720:
                        tmp_crop[crop_size*2-cr_y:,:cr_x] = crop
                        tmp_mask1[crop_size*2-cr_y:,:cr_x] = crop_mask1
                        tmp_mask2[crop_size*2-cr_y:,:cr_x] = crop_mask2
                    else:
                        tmp_crop[:cr_y,:cr_x] = crop
                        tmp_mask1[:cr_y,:cr_x] = crop_mask1
                        tmp_mask2[:cr_y,:cr_x] = crop_mask2

                elif x2> 1280:
                    if y2>=720:
                        tmp_crop[crop_size*2-cr_y:,crop_size*2-cr_x:] = crop
                        tmp_mask1[crop_size*2-cr_y:,crop_size*2-cr_x:] = crop_mask1
                        tmp_mask2[crop_size*2-cr_y:,crop_size*2-cr_x:] = crop_mask2
                    else:
                        tmp_crop[:cr_y,crop_size*2-cr_x:] = crop
                        tmp_mask1[:cr_y,crop_size*2-cr_x:] = crop_mask1
                        tmp_mask2[:cr_y,crop_size*2-cr_x:] = crop_mask2

                tmp_crop = cv2.resize(tmp_crop, (out_size*2,out_size*2))
                tmp_mask1 = cv2.resize(tmp_mask1, (out_size*2,out_size*2))
                tmp_mask2 = cv2.resize(tmp_mask2, (out_size*2,out_size*2))
                images.append(tmp_crop)
                masks1.append(tmp_mask1)
                masks2.append(tmp_mask2)
        else:
            empty_count +=1
            # crop =  np.zeros((crop_size*2, crop_size*2,3))
            crop =  np.zeros((out_size*2, out_size*2,3), dtype = np.uint8)
            crop_mask1 =  np.zeros((out_size*2, out_size*2), dtype = np.uint8)
            crop_mask2 =  np.zeros((out_size*2, out_size*2), dtype = np.uint8)
            images.append(crop)
            masks1.append(crop_mask1)
            masks2.append(crop_mask2)
    return images, masks1, masks2

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dict, tfms=None, cfg=None):
        self.df = df.reset_index(drop=True)
        self.image_dict = image_dict
        self.transform = A.ReplayCompose([
        A.Resize(cfg.img_size, cfg.img_size, interpolation=1, p=1),
    ])
        self.cfg = cfg
        
        self.trk_step = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        idx1 = int(row['nfl_player_id_1'])
        if self.cfg.is_G:
            # idx2 = row['nfl_player_id_2']
            idx2 = 'G'
        else:
            idx2 = int(row['nfl_player_id_2'])
        fr_id = int(row['frame'])
        step = int(row['step'])
        
        if self.cfg.is_G:
            e_images, e_masks1, _ = get_sample_G(e_vid, fr_id, idx1, idx2, window_size = window_size, stride=4)
            s_images, s_masks1, _ = get_sample_G(s_vid, fr_id, idx1, idx2, window_size = window_size, stride=4, out_size = 128)
        else:
            e_images, e_masks1, e_masks2 = get_sample(e_vid, fr_id, idx1, idx2, window_size = window_size, stride=4)
            s_images, s_masks1, s_masks2 = get_sample(s_vid, fr_id, idx1, idx2, window_size = window_size, stride=4, out_size = 128)
        
        images_e = []
        images_s = []
        for ii in range(len(e_images)):
            e_img = e_images[ii]
            e_img[e_masks1[ii]>100] = 255
            if not  self.cfg.is_G:
                e_img[e_masks2[ii]>100] = 0

            s_img = s_images[ii]
            s_img[s_masks1[ii]>100] = 255
            if not self.cfg.is_G:
                s_img[s_masks2[ii]>100] = 0

            images_e.append(e_img)
            images_s.append(s_img)


        e_images = np.array(images_e)
        s_images = np.array(images_s)
        
        
        num_empty = 0
        for img in e_images:
            h, w, c = img.shape 
            if np.sum(img<2)/(h*w*c) > 0.9:
                num_empty += 1
        if len(e_images) - num_empty < 2:
            # print('e empty')
            e_images = s_images.copy()

        num_empty = 0
        for img in s_images:
            h, w, c = img.shape 
            if np.sum(img<2)/(h*w*c) > 0.9:
                num_empty += 1
        if len(s_images) - num_empty < 2:
            # print('s empty')
            s_images = e_images.copy()
            
        
        if self.trk_step == 0:
            self.trk_step = len(e_images)//2

        if not self.cfg.is_G:
            trk_images = self.render_trk(vid, step, idx1, idx2)
            
        images = []
        for i in range(len(s_images)):
            if not self.cfg.is_G:
                trk_img = trk_images[i]
                img = np.hstack([e_images[i], trk_img, s_images[i]])
            else:
                img = np.hstack([e_images[i], s_images[i]])
            images.append(img)
        img = np.array(images)
        img = img.transpose(3,0,1,2) #C T H W

        img = img/255

        return torch.tensor(img, dtype=torch.float)
    
    def render_trk(self, vid, step, idx1, idx2):
        shift_x = 0
        shift_y = 0

        d_x = 5
        scale = 60/d_x

        idx = f'{vid}_{step}'
        images = []
        x1 = trk_dict[idx][idx1]['x']
        y1 = trk_dict[idx][idx1]['y']

        x2 = trk_dict[idx][idx2]['x']
        y2 = trk_dict[idx][idx2]['y']

        xc = 0.5*x1 + 0.5*x2
        yc = 0.5*y1 + 0.5*y2

        for st in range(step-self.trk_step, step + self.trk_step + 1):
            #print('=======',step, self.trk_step)
            this_idx = f'{vid}_{st}'
            img = np.zeros((3, self.cfg.img_size, 128), dtype=np.uint8)
            if this_idx in trk_dict:
                for p_id, meta in trk_dict[this_idx].items():
                    x = meta['x']
                    y = meta['y']
                    t = meta['t']

                    x = x - xc + d_x
                    y = y - yc + (2*d_x)

                    x = round(x*scale) + shift_x
                    y = round(y*scale) + shift_y

                    if x>0 and y>0 and x<128 and y<self.cfg.img_size:
                        radius = 3
                        val = 125
                        if p_id in [idx1, idx2]:
                            radius = 5
                            val = 255

                        cv2.circle(img[0], (x, y), radius, val, thickness=-1)
                        if t == 'home':
                            cv2.circle(img[1], (x, y), radius, val, thickness=-1)
                        else:
                            cv2.circle(img[2], (x, y), radius, val, thickness=-1)


            img = img.transpose(1,2,0)
            images.append(img)

        return images



def predict_p(df_g, vid):
    if df_g.shape[0] == 0:
        return df_g, 0
    # cfg.is_G = 0
    test_dataset = TestDataset(df_g, image_dict, cfg=cfg)
    test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=8,
            num_workers=8,
            shuffle=False)
    y_preds = []
    with torch.no_grad():
        bar = tqdm(test_loader)
        for batch_idx, images in enumerate(bar):
            images = images.float().to(device)
            with torch.cuda.amp.autocast():
                pred = model(images)
                logit = pred['out1'].sigmoid().detach().cpu().numpy()
                out = logit
            y_preds.append(out)
    y_preds = np.concatenate(y_preds)
    print(y_preds.shape, df_g.shape)
    df_g['pred'] = y_preds
    return df_g, 1

frame_shift = 6
window_size = [-44, -37, -30, -24, -18, -13, -8, -4, -2, 0, 2, 4, 8, 13, 18, 24, 30, 37]
window_size = [x+frame_shift for x in window_size]

class config:
    model_name = 'r50ir' 
    pool_type = 'avg'
    is_G = 0
    img_size = 256
    name = 'r50ir_csn_c15_m1_d2_all'
    weight_dir = 'outputs/'

cfg = config

df = pd.read_csv('../tree/pre_p_xgb/model/xgb_oof_pair_fe.csv')
df = df[df.pred>0.005]

df['vid'] = df['contact_id'].apply(lambda x: '_'.join(x.split('_')[:2]))
df['step'] = df['contact_id'].apply(lambda x: int(x.split('_')[2]))


df = df.reset_index(drop=True)         
print(df.shape)

det_dict = np.load('../data/det_dict.npy', allow_pickle=True).item()
trk_dict = np.load('../data/trk_pos.npy', allow_pickle=True).item()
device = "cuda"

for fold in [0,1,2,3,4]:
    ckpt_path = f'{cfg.weight_dir}/{cfg.name}/{cfg.name}_last_f{fold}.pth'
    # ckpt_path = f'{cfg.weight_dir}/{cfg.name}/swa_best_f{fold}.pth'
    model = NModel(cfg)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    this_df = df[df.fold==fold]

    print(f'process fold {fold}', this_df.shape)

    vids = this_df.vid.unique()

    results = []
    for vid in tqdm(vids):
        print(f'process vid {vid}')
        e_vid = vid + f'_Endzone'
        s_vid = vid + f'_Sideline'
        e_vid_path = f'../data/train/{e_vid}.mp4'
        s_vid_path = f'../data/train/{s_vid}.mp4'
        
        image_dict = {}
        
        cap = cv2.VideoCapture(e_vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES,250)
        frame_count = 250
        while 1:
            ret, frame = cap.read()
            if not ret:
                break
            kk = f'{e_vid}_{frame_count}'
            image_dict[kk] = frame
            frame_count +=1

        cap = cv2.VideoCapture(s_vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 250)
        frame_count = 250
        while 1:
            ret, frame = cap.read()
            if not ret:
                break
            kk = f'{s_vid}_{frame_count}'
            image_dict[kk] = frame
            frame_count +=1
        
        this_vid_df = this_df[this_df.vid==vid]
        df_p, nona_p = predict_p(this_vid_df, vid)
        # print(df_p.shape, nona_p)
        if nona_p:
            results.append(df_p)
        gc.collect()

    out_df = pd.concat(results)
    out_df = out_df[['contact_id', 'pred']]
    out_df.to_csv(f'{cfg.weight_dir}/{cfg.name}/oof_f{fold}_s{frame_shift}_xgb.csv', index=False)
