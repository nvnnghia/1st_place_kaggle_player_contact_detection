import albumentations as A
import cv2
from default_config import basic_cfg
import os

cfg = basic_cfg

cfg.debug = 0
cfg.batch_size = 6
cfg.out_dir = f"outputs/{os.path.basename(__file__).split('.')[0]}"
cfg.train_csv_path = "train_cache15_G.csv"
cfg.seed = 48
cfg.epochs = 1
cfg.mode = 'train'
cfg.model_name = 'r50ir'
cfg.lr = 4e-5
cfg.scheduler = "linear" #linear  cosine, step cosinewarmup warmupv2
cfg.optimizer = "Adam"  # Adam, SGD, AdamW
cfg.num_workers = 8
cfg.folds = [0,1,2,3,4,6,7,8]
cfg.model = 'model_csn1'
cfg.dataset = 'dataset_3d_3ch_v2'
cfg.drop_rate = 0.15
cfg.img_size = 256
cfg.use_swa = 0
cfg.frac = 3
cfg.pos_frac = 1
cfg.val_frac = 1.5
cfg.is_G = 1
# cfg.sampler = 1

cfg.load_weight = 'pretrained/vmz_ircsn_ig65m_pretrained_r50_32x2x1_58e_kinetics400_rgb_20210617-86d33018.pth'

base_aug = [
        A.OneOf([
            A.RandomGamma(gamma_limit=(30, 150), p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=1),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            A.CLAHE(clip_limit=5.0, tile_grid_size=(5, 5), p=1),
        ], p=0.8),

        # A.OneOf([
        #    A.ElasticTransform(always_apply=False, p=1.0, alpha=1.0, sigma=15, alpha_affine=15, interpolation=1, border_mode=0, value=125, mask_value=None, approximate=False),
        #    A.GridDistortion(always_apply=False, p=1.0, num_steps=2, distort_limit=(-0.2, 0.2), interpolation=1, border_mode=0, value=(0, 0, 0), mask_value=None),
        #    A.Perspective(always_apply=False, p=1.0, scale=(0.05, 0.1), keep_size=1, pad_mode=0, pad_val=(0, 0, 0), mask_pad_val=0, fit_output=0, interpolation=1),
        # ], p=0.1),

        A.HorizontalFlip(p=0.5), 
        # A.VerticalFlip(p=0.5),
        # A.Transpose(p=0.5),

        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.1, rotate_limit=15,
                                        interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        A.Cutout(max_h_size=int(50), max_w_size=int(50), num_holes=2, p=0.5),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ]

e_resize = [A.RandomResizedCrop(always_apply=False, p=1.0, height=cfg.img_size, width=cfg.img_size, scale=(0.7, 1.2), ratio=(0.75, 1.3), interpolation=1)]
s_resize = [A.RandomResizedCrop(always_apply=False, p=1.0, height=cfg.img_size, width=cfg.img_size, scale=(0.7, 1.2), ratio=(0.75, 1.3), interpolation=1)]

cfg.train_e_transform = A.ReplayCompose(e_resize+base_aug)
cfg.train_s_transform = A.ReplayCompose(s_resize+base_aug)


cfg.val_e_transform = A.ReplayCompose([
        A.Resize(cfg.img_size, cfg.img_size, interpolation=1, p=1),
    ])

cfg.val_s_transform = A.ReplayCompose([
        A.Resize(cfg.img_size, cfg.img_size, interpolation=1, p=1),
    ])