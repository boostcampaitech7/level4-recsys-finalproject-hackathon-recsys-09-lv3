#!/bin/bash

# --pretrain_model_path pretrain_backbones 지정하여 backbone 모델 저장 및 불러오기

##################################### STEAM ##########################################
######### steam * deepfm #########
# Dataset: steam Backbone: deepfm Warmup_model: None
python main.py --dataset_name steam --model_name deepfm  --warmup_model base --pretrain_model_path pretrain_backbones
# Dataset: steam Backbone: deepfm Warmup_model: DropoutNet 
python main.py --dataset_name steam --model_name deepfm  --warmup_model base  --is_dropoutnet True --pretrain_model_path pretrain_backbones
# Dataset: steam Backbone: deepfm Warmup_model: Meta-Embedding
python main.py --dataset_name steam --model_name deepfm  --warmup_model metaE --pretrain_model_path pretrain_backbones
# Dataset: steam Backbone: deepfm Warmup_model: MWUF
python main.py --dataset_name steam --model_name deepfm  --warmup_model mwuf --pretrain_model_path pretrain_backbones
# Dataset: steam Backbone: deepfm Warmup_model: CVAR(Init Only)
python main.py --dataset_name steam --model_name deepfm  --warmup_model cvar_init --cvar_iters 10 --pretrain_model_path pretrain_backbones
# Dataset: steam Backbone: deepfm Warmup_model: CVAR
python main.py --dataset_name steam --model_name deepfm  --warmup_model cvar --cvar_iters 10 --pretrain_model_path pretrain_backbones


##################################### Data Points in Table 1 ##########################################
######### movielens1M * deepfm #########
# Dataset: movielens1M Backbone: deepfm Warmup_model: None
python main.py --dataset_name movielens1M --model_name deepfm  --warmup_model base --pretrain_model_path pretrain_backbones
# Dataset: movielens1M Backbone: deepfm Warmup_model: DropoutNet 
python main.py --dataset_name movielens1M --model_name deepfm  --warmup_model base  --is_dropoutnet True --pretrain_model_path pretrain_backbones
# Dataset: movielens1M Backbone: deepfm Warmup_model: Meta-Embedding
python main.py --dataset_name movielens1M --model_name deepfm  --warmup_model metaE --pretrain_model_path pretrain_backbones
# Dataset: movielens1M Backbone: deepfm Warmup_model: MWUF
python main.py --dataset_name movielens1M --model_name deepfm  --warmup_model mwuf --pretrain_model_path pretrain_backbones
# Dataset: movielens1M Backbone: deepfm Warmup_model: CVAR(Init Only)
python main.py --dataset_name movielens1M --model_name deepfm  --warmup_model cvar_init --cvar_iters 10 --pretrain_model_path pretrain_backbones
# Dataset: movielens1M Backbone: deepfm Warmup_model: CVAR
python main.py --dataset_name movielens1M --model_name deepfm  --warmup_model cvar --cvar_iters 10 --pretrain_model_path pretrain_backbones

######### movielens1M * wide&deep #########
# Dataset: movielens1M Backbone: wide&deep Warmup_model: None
python main.py --dataset_name movielens1M --model_name wd  --warmup_model base 
# Dataset: movielens1M Backbone: wide&deep Warmup_model: DropoutNet
python main.py --dataset_name movielens1M --model_name wd  --warmup_model base  --is_dropoutnet True
# Dataset: movielens1M Backbone: wide&deep Warmup_model: Meta-Embedding
python main.py --dataset_name movielens1M --model_name wd  --warmup_model metaE
# Dataset: movielens1M Backbone: wide&deep Warmup_model: MWUF
python main.py --dataset_name movielens1M --model_name wd  --warmup_model mwuf
# Dataset: movielens1M Backbone: wide&deep Warmup_model: CVAR(Init Only)
python main.py --dataset_name movielens1M --model_name wd  --warmup_model cvar_init --cvar_iters 10
# Dataset: movielens1M Backbone: wide&deep Warmup_model: CVAR
python main.py --dataset_name movielens1M --model_name wd  --warmup_model cvar --cvar_iters 10

######### taobaoAD * deepfm #########
# Dataset: taobaoAD Backbone: deepfm Warmup_model: None
python main.py --dataset_name taobaoAD --model_name deepfm  --warmup_model base 
# Dataset: taobaoAD Backbone: deepfm Warmup_model: DropoutNet
python main.py --dataset_name taobaoAD --model_name deepfm  --warmup_model base  --is_dropoutnet True
# Dataset: taobaoAD Backbone: deepfm Warmup_model: Meta-Embedding
python main.py --dataset_name taobaoAD --model_name deepfm  --warmup_model metaE
# Dataset: taobaoAD Backbone: deepfm Warmup_model: MWUF
python main.py --dataset_name taobaoAD --model_name deepfm  --warmup_model mwuf
# Dataset: taobaoAD Backbone: deepfm Warmup_model: CVAR(Init Only)
python main.py --dataset_name taobaoAD --model_name deepfm  --warmup_model cvar_init --cvar_iters 1
# Dataset: taobaoAD Backbone: deepfm Warmup_model: CVAR
python main.py --dataset_name taobaoAD --model_name deepfm  --warmup_model cvar --cvar_iters 1

######### taobaoAD * wd #########
# Dataset: taobaoAD Backbone: wide&deep Warmup_model: None
python main.py --dataset_name taobaoAD --model_name wd  --warmup_model base 
# Dataset: taobaoAD Backbone: wide&deep Warmup_model: DropoutNet
python main.py --dataset_name taobaoAD --model_name wd  --warmup_model base  --is_dropoutnet True
# Dataset: taobaoAD Backbone: wide&deep Warmup_model: Meta-Embedding
python main.py --dataset_name taobaoAD --model_name wd  --warmup_model metaE
# Dataset: taobaoAD Backbone: wide&deep Warmup_model: MWUF
python main.py --dataset_name taobaoAD --model_name wd  --warmup_model mwuf
# Dataset: taobaoAD Backbone: wide&deep Warmup_model: CVAR(Init Only)
python main.py --dataset_name taobaoAD --model_name wd  --warmup_model cvar_init --cvar_iters 1
# Dataset: taobaoAD Backbone: wide&deep Warmup_model: CVAR
python main.py --dataset_name taobaoAD --model_name wd  --warmup_model cvar --cvar_iters 1

##################################### Data Points in Figure 1 ##########################################
######### movielens1M * [fm, deepfm, wd, dcn, ipnn, opnn] #########
# Dataset: movielens1M Backbone: fm Warmup_model: None
python main.py --dataset_name movielens1M --model_name fm  --warmup_model base 
# Dataset: movielens1M Backbone: fm Warmup_model: cvar
python main.py --dataset_name movielens1M --model_name fm  --warmup_model cvar --cvar_iters 10

# Dataset: movielens1M Backbone: deepfm Warmup_model: None
python main.py --dataset_name movielens1M --model_name deepfm  --warmup_model base 
# Dataset: movielens1M Backbone: deepfm Warmup_model: cvar
python main.py --dataset_name movielens1M --model_name deepfm  --warmup_model cvar --cvar_iters 10

# Dataset: movielens1M Backbone: wide&deep Warmup_model: None
python main.py --dataset_name movielens1M --model_name wd  --warmup_model base 
# Dataset: movielens1M Backbone: wide&deep Warmup_model: cvar
python main.py --dataset_name movielens1M --model_name wd  --warmup_model cvar --cvar_iters 10

# Dataset: movielens1M Backbone: dcn Warmup_model: None
python main.py --dataset_name movielens1M --model_name dcn  --warmup_model base 
# Dataset: movielens1M Backbone: dcn Warmup_model: cvar
python main.py --dataset_name movielens1M --model_name dcn  --warmup_model cvar --cvar_iters 10

# Dataset: movielens1M Backbone: ipnn Warmup_model: None
python main.py --dataset_name movielens1M --model_name ipnn  --warmup_model base 
# Dataset: movielens1M Backbone: ipnn Warmup_model: cvar
python main.py --dataset_name movielens1M --model_name ipnn  --warmup_model cvar --cvar_iters 10

# Dataset: movielens1M Backbone: opnn Warmup_model: None
python main.py --dataset_name movielens1M --model_name opnn  --warmup_model base 
# Dataset: movielens1M Backbone: opnn Warmup_model: cvar
python main.py --dataset_name movielens1M --model_name opnn  --warmup_model cvar --cvar_iters 10

######### taobaoAD * [fm, deepfm, wd, dcn, ipnn, opnn] #########
# Dataset: taobaoAD Backbone: fm Warmup_model: None
python main.py --dataset_name taobaoAD --model_name fm  --warmup_model base 
# Dataset: taobaoAD Backbone: fm Warmup_model: cvar
python main.py --dataset_name taobaoAD --model_name fm  --warmup_model cvar --cvar_iters 1

# Dataset: taobaoAD Backbone: deepfm Warmup_model: None
python main.py --dataset_name taobaoAD --model_name deepfm  --warmup_model base 
# Dataset: taobaoAD Backbone: deepfm Warmup_model: cvar
python main.py --dataset_name taobaoAD --model_name deepfm  --warmup_model cvar --cvar_iters 1

# Dataset: taobaoAD Backbone: wide&deep Warmup_model: None
python main.py --dataset_name taobaoAD --model_name wd  --warmup_model base 
# Dataset: taobaoAD Backbone: wide&deep Warmup_model: cvar
python main.py --dataset_name taobaoAD --model_name wd  --warmup_model cvar --cvar_iters 1

# Dataset: taobaoAD Backbone: dcn Warmup_model: None
python main.py --dataset_name taobaoAD --model_name dcn  --warmup_model base 
# Dataset: taobaoAD Backbone: dcn Warmup_model: cvar
python main.py --dataset_name taobaoAD --model_name dcn  --warmup_model cvar --cvar_iters 1

# Dataset: taobaoAD Backbone: ipnn Warmup_model: None
python main.py --dataset_name taobaoAD --model_name ipnn  --warmup_model base 
# Dataset: taobaoAD Backbone: ipnn Warmup_model: cvar
python main.py --dataset_name taobaoAD --model_name ipnn  --warmup_model cvar --cvar_iters 1

# Dataset: taobaoAD Backbone: opnn Warmup_model: None
python main.py --dataset_name taobaoAD --model_name opnn  --warmup_model base 
# Dataset: taobaoAD Backbone: opnn Warmup_model: cvar
python main.py --dataset_name taobaoAD --model_name opnn  --warmup_model cvar --cvar_iters 1



