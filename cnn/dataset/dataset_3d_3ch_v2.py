import torch 
import pandas as pd 
from torch.utils.data import Dataset
from albumentations import ReplayCompose
from torch.utils.data.sampler import Sampler
import numpy as np 
from tqdm import tqdm
import torchvision
import cv2 
import os
import gc
import random 
# random.seed(42)

class SimpleClassSampler(Sampler):
    def __init__(self, df, cfg):
        self.cfg=cfg
        self.df = df.reset_index(drop=True)
        self.index_class1 = self.df[self.df.contact==1].index.to_list()
        self.index_class0 = self.df[self.df.contact==0].index.to_list()
        
        self.length = int(self.cfg.pos_frac*(len(self.index_class1))) + int(self.cfg.frac*(len(self.index_class0)))

    def __iter__(self):
        random_choice1 = np.random.choice(self.index_class1, int(self.cfg.pos_frac*(len(self.index_class1))), replace=False)

        random_choice0 = np.random.choice(self.index_class0, int(self.cfg.frac*(len(self.index_class0))), replace=False)

        print('======',len(random_choice0), len(random_choice1))
        # print('======',random_choice0[:10], random_choice1[:10])
        
        all_indexs = list(random_choice0) + list(random_choice1)

        l = np.array(all_indexs)
        l = l.reshape(-1)
        random.shuffle(l)
        return iter(l)

    def __len__(self):
        return int(self.length)

class NDataset(Dataset):
    def __init__(self, cfg, df, tfms=None, fold_id = 0, is_train = True):
        super().__init__()
        self.df = df.reset_index(drop=True)#.sample(frac = 1.0, random_state=42) 
        self.cfg = cfg
        self.fold_id = fold_id
        self.transform = tfms
        self.is_train = is_train
        self.feat_cols = ['x_position_1', 'y_position_1', 'distance']
        self.trk_step = 0
        if self.is_train:
            self.e_transform = self.cfg.train_e_transform
            self.s_transform = self.cfg.train_s_transform
        else:
            self.e_transform = self.cfg.val_e_transform
            self.s_transform = self.cfg.val_s_transform
        self.trk_dict = np.load('../data/trk_pos.npy', allow_pickle=True).item()

        print(f'Fold: {fold_id}, is_train: {is_train}, total frame {len(self.df)}')

    def __getitem__(self, index):
        row = self.df.loc[index]
        path = row['path']
        step = row['step']
        idx = path.split('/')[-1]
        vid = '_'.join(idx.split('_')[:2])

        if '_ext' in path:
            idx1 = idx.split('_')[2]
        else:
            idx1 = int(idx.split('_')[2])
            
        if not self.cfg.is_G:
            if '_ext' in path:
                idx2 = idx.split('_')[3]
            else:
                idx2 = int(idx.split('_')[3])

        e_path = f'{path}_e.npy'
        s_path = f'{path}_s.npy'

        e_images = np.load(e_path)
        s_images = np.load(s_path)

        if self.cfg.skip_frame > 0:
            e_images = e_images[self.cfg.skip_frame:-self.cfg.skip_frame,:,:,:]
            s_images = s_images[self.cfg.skip_frame:-self.cfg.skip_frame,:,:,:]

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
            trk_images = self.render_trk(vid, step, idx1, idx2, self.trk_dict)

        replay = None
        e_images_ = []
        for img in e_images:
            if replay is None:
                sample = self.e_transform(image=img)
                replay = sample["replay"]
            else:
                sample = ReplayCompose.replay(replay, image=img)
            img = sample["image"]
            e_images_.append(img)

        replay = None
        s_images_ = []
        for img in s_images:
            if replay is None:
                sample = self.s_transform(image=img)
                replay = sample["replay"]
            else:
                sample = ReplayCompose.replay(replay, image=img)
            img = sample["image"]
            s_images_.append(img)

        #simple trk image augmentation
        flip_trk_lr = False
        flip_trk_ud = False
        is_swap = False
        if self.is_train:
            if random.random() < 0.5:
                flip_trk_lr = True
            if random.random() < 0.5:
                flip_trk_ud = True

            if random.random() < 0.5:
                is_swap = True

        images = []
        for i in range(len(s_images)):

            if not self.cfg.is_G:
                trk_img = trk_images[i]
                if flip_trk_lr:
                    trk_img = np.fliplr(trk_img)
                if flip_trk_ud:
                    trk_img = np.flipud(trk_img)

                # s_img = np.vstack([trk_img, s_images_[i]])
                if is_swap:
                    img = np.hstack([s_images_[i], trk_img, e_images_[i]])
                else:
                    img = np.hstack([e_images_[i], trk_img, s_images_[i]])
            else:
                if is_swap:
                    img = np.hstack([s_images_[i], e_images_[i]])
                else:
                    img = np.hstack([e_images_[i], s_images_[i]])

            images.append(img)

        
        img = np.array(images)
        # print(img.shape)
        if self.cfg.model in ['model_25d']:
            # img = img.transpose(0,3,1,2)
            img = np.concatenate(img, axis=2)
            img = img.transpose(2,0,1)
        else:
            img = img.transpose(3,0,1,2) #C T H W

        img = img/255

        img = torch.from_numpy(img)

        target = row['contact']

        if self.cfg.use_meta:
            feat = row[self.feat_cols]
        else:
            feat = target

        if self.cfg.use_oof and self.is_train:
            feat = row['pred']

        # print(img.shape, mask.shape)

        return torch.tensor(img, dtype=torch.float), torch.tensor(target, dtype=torch.float), torch.tensor(feat, dtype=torch.float)

    def render_trk(self, vid, step, idx1, idx2, trk_dict):
        if self.is_train:
            shift_x = random.randint(-10,10)
            shift_y = random.randint(-20,20)
        else:
            shift_x = 0
            shift_y = 0

        # d_x = 0.1*random.randint(30,60)
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
                        if self.cfg.trk_type == 1:
                            #v1
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

                        elif self.cfg.trk_type == 2:
                            ##v2
                            radius = 4
                            val = 255
                            if p_id in [idx1, idx2]:
                                radius = 6
                                val = 255

                            # if p_id in [idx1, idx2]:
                            #     cv2.circle(img[0], (x, y), radius, val, thickness=-1)
                            if t == 'home':
                                cv2.circle(img[1], (x, y), radius, val, thickness=-1)
                            else:
                                cv2.circle(img[2], (x, y), radius, val, thickness=-1)

            img = img.transpose(1,2,0)
            images.append(img)

        return images

    def __len__(self):
        return len(self.df) 

