#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py -C r50ir_csn_c11_m1_d2_G_all   
CUDA_VISIBLE_DEVICES=0 python train.py -C r50ir_csn_c15_m1_d2_G_all  

CUDA_VISIBLE_DEVICES=0 python train.py -C r50ir_csn_c11_m1_d2_G_all  -M val
CUDA_VISIBLE_DEVICES=0 python train.py -C r50ir_csn_c15_m1_d2_G_all  -M val

CUDA_VISIBLE_DEVICES=0 python train.py -C r50ir_csn_c15_m1_d2_all  
CUDA_VISIBLE_DEVICES=0 python inference.py
