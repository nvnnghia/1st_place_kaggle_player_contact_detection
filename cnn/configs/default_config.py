import os
from types import SimpleNamespace
import albumentations

cfg = SimpleNamespace(**{})

cfg.debug = 0
cfg.train_csv_path = "train_cache1.csv"
cfg.batch_size = 64
cfg.out_dir = f"/data2/weights/nfl/sed3d/{os.path.basename(__file__).split('.')[0]}"
cfg.seed = 42
cfg.img_size = 256
cfg.epochs = 20
cfg.mode = 'train'
cfg.model_name = 'tf_efficientnet_b0_ns'
cfg.lr = 1e-3
cfg.scheduler = "linear" #linear  cosine, step cosinewarmup warmupv2
cfg.optimizer = "Adam"  # Adam, SGD, AdamW
cfg.num_workers = 0
cfg.folds = [0,1,2,3,4]
cfg.model = 'model1'
cfg.dataset = 'dataset1'
cfg.loss_fn  = 'bce'
cfg.load_weight = ''
cfg.apex=False
cfg.use_meta = 0
cfg.frac = 0.25
cfg.val_frac = 1
cfg.pos_frac = 1
cfg.pool_type = 'avg'
cfg.trk_type = 1
cfg.num_freeze = 0
cfg.hoge = 0
cfg.use_swa = 0
cfg.warmup = 500 
cfg.is_G = 0
cfg.sampler = 0
cfg.skip_frame = 0
cfg.use_oof = 0

cfg.train_transform = albumentations.Compose([
        # albumentations.RandomResizedCrop(always_apply=False, p=1.0, height=cfg.img_size, width=cfg.img_size, scale=(0.7, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=0),
        # albumentations.Resize(cfg.img_size, cfg.img_size, interpolation=1, p=1),
        # albumentations.OneOf([
        #     albumentations.RandomGamma(gamma_limit=(60, 120), p=0.9),
        #     albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
        #     albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
        # ]),
        # albumentations.OneOf([
        #     albumentations.Blur(blur_limit=4, p=1),
        #     albumentations.MotionBlur(blur_limit=4, p=1),
        #     albumentations.MedianBlur(blur_limit=3, p=1)
        # ], p=0.5),
        # albumentations.HorizontalFlip(p=0.5), 
        albumentations.VerticalFlip(p=0.5),
        albumentations.Transpose(p=0.5),
        albumentations.Rotate(always_apply=False, p=0.5, limit=(-90, 90), interpolation=1, border_mode=0, value=(127, 127, 127), mask_value=None)

        # albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20,
        #                                 interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=1),
        # albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ])
cfg.val_transform = albumentations.Compose([
        # albumentations.Resize(cfg.img_size, cfg.img_size, interpolation=1, p=1),
        # albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ])

basic_cfg = cfg
