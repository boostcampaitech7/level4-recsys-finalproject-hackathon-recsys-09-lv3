#!/bin/bash

# --pretrain_model_path pretrain_backbones 지정하여 backbone 모델 저장 및 불러오기

##################################### STEAM ##########################################
######### steam * deepfm #########

# Dataset: steam Backbone: deepfm Warmup_model: None
nohup python main.py --dataset_name steam --model_name deepfm --warmup_model base --pretrain_model_path pretrain_backbones > nohup_steam_base.log 2>&1 &
wait  # 해당 프로세스가 끝날 때까지 대기

nohup python main.py --dataset_name steam --model_name deepfm --warmup_model deepfm_only --pretrain_model_path pretrain_backbones > nohup_steam_base.log 2>&1 &


# Dataset: steam Backbone: deepfm Warmup_model: DropoutNet 
#nohup python main.py --dataset_name steam --model_name deepfm --warmup_model base --is_dropoutnet True --pretrain_model_path pretrain_backbones > nohup_steam_dropoutnet.log 2>&1 &
#wait  

# Dataset: steam Backbone: deepfm Warmup_model: Meta-Embedding
nohup python main.py --dataset_name steam --model_name deepfm --warmup_model metaE --pretrain_model_path pretrain_backbones > nohup_steam_metaE.log 2>&1 &
wait  

# Dataset: steam Backbone: deepfm Warmup_model: MWUF
nohup python main.py --dataset_name steam --model_name deepfm --warmup_model mwuf --pretrain_model_path pretrain_backbones > nohup_steam_mwuf.log 2>&1 &
wait  

# Dataset: steam Backbone: deepfm Warmup_model: CVAR(Init Only)
#nohup python main.py --dataset_name steam --model_name deepfm --warmup_model cvar_init --cvar_iters 10 --pretrain_model_path pretrain_backbones > nohup_steam_cvar_init.log 2>&1 &
#wait  

# Dataset: steam Backbone: deepfm Warmup_model: CVAR
nohup python main.py --dataset_name steam --model_name deepfm --warmup_model cvar --cvar_iters 10 --pretrain_model_path pretrain_backbones > nohup_steam_cvar.log 2>&1 &
wait

'''## steam_500_15

nohup python main.py --dataset_name steam_500_15 --model_name deepfm --warmup_model base --pretrain_model_path pretrain_backbones > nohup_steam_500_15_base.log 2>&1 &
wait  # 해당 프로세스가 끝날 때까지 대기

nohup python main.py --dataset_name steam_500_15 --model_name deepfm --warmup_model metaE --pretrain_model_path pretrain_backbones > nohup_steam_500_15_metaE.log 2>&1 &
wait  

nohup python main.py --dataset_name steam_500_15 --model_name deepfm --warmup_model mwuf --pretrain_model_path pretrain_backbones > nohup_steam_500_15_mwuf.log 2>&1 &
wait  

#nohup python main.py --dataset_name steam_500_15 --model_name deepfm --warmup_model cvar_init --cvar_iters 10 --pretrain_model_path pretrain_backbones > nohup_steam_cvar_init.log 2>&1 &
#wait  

nohup python main.py --dataset_name steam_500_15 --model_name deepfm --warmup_model cvar --cvar_iters 10 --pretrain_model_path pretrain_backbones > nohup_steam_500_15_cvar.log 2>&1 &
wait  '''
