# 1st and Future - Player Contact Detection

Below you can find an outline of how to reproduce my 1st place solution for the `1st and Future - Player Contact Detection`.

## 1.INSTALLATION
- Ubuntu 18.04.5 LTS
- CUDA 11.2
- Python 3.7.5
- Training PC: 1x RTX3090 (or any GPU with at least 24Gb VRAM), 32GB RAM, at least 5TB of disk space.
- python packages are detailed separately in requirements.txt
```
$ conda create -n envs python=3.7.5
$ conda activate envs
$ pip install -r requirements.txt
$ pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
$ pip install -U openmim
$ mim install mmcv-full

```

## 2.DATA
* Download dataset and extract to `data/` folder.
* Download pretrained mmaction2 weight `wget -P pretrained/ "https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ircsn_ig65m_pretrained_r50_32x2x1_58e_kinetics400_rgb_20210617-86d33018.pth"`

## 3. PREPROCESSING
* Create folds and cache helmet boxes and tracking data to dictionary.  
```
$ python create_folds.py
$ python cache_metadata.py
```
* Train xgb preprocessing to filter easy negative sample.  
```
$ cd tree/
$ python pre_g_xgb.py
$ python pre_p_xgb.py
```
* Generate cache for CNN model and save to disk.  
```
$ cd cnn/
$ python create_cache11_G.py
$ python create_cache15_G.py
$ python create_cache15.py
```

* The project folder now looks like  
├── data   
│ ├── train    
│ ├── test    
│ ├── train_player_tracking.csv   
│ ├── train_folds.csv  
│ ├── ...  
├── tree   
│ ├── pp_g_xgb.py    
│ ├── pp_p_xgb.py    
│ ├── pre_g_xgb  
│ ├── ...  
├── cnn   
│ ├── configs   
│ │ ├── r50ir_csn_c11_m1_d2_G_all.py  
│ │ ├── r50ir_csn_c15_m1_d2_all.py   
│ │ ├── ...   
│ ├── models   
│ │ ├── model_csn1.py  
│ │ ├── resnet3d_csn.py   
│ ├── dataset   
│ │ ├── dataset_3d_3ch_v2.py  
│ ├── cache     
│ │ ├── cache11_G  
│ │ ├── cache15_G  
│ │ ├── cache15  
│ ├── pretrained/   
│ ├── train.py   
│ ├── inference.py   
│ ├── create_cache11_G.py   
│ ├── create_cache15_G.py   
│ ├── create_cache15.py   
│ ├── train.sh  
│ ├── ...   
├── cache_metadata.py  
├── create_folds.py  
├── readme.md  
├── requirements.txt  

## 4.TRAINING
* Preprocessing step must be run before this step.
* To train and validate all the models, run the following command inside the `cnn/` folder. 
```
$ ./train.sh
```
   - Folds 0,1,2,3,4 are used for validation and train a xgb post processing model. 
   - Folds 5,6,7,8,9 are trained with all data and used in submission. PP model used fold 5,6,7,8; PG model used fold 6,7,8 from 2 configs (6 checkpoints).

* Train xgb post processing.  
```
$ cd tree/
$ python pp_g_xgb.py
$ python pp_p_xgb.py
```

## 5.INFERENCE
* Since this is a code competition, the inference code is available in kaggle notebook.
* https://www.kaggle.com/code/nvnnghia/nfl3-ensemble-w-pp-sub?scriptVersionId=121109062
