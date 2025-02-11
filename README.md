# 🍀루키비키자나 TVING 해커톤

Overview
--------
추천 시스템은 사용자 이력을 바탕으로 사용자가 선호하는 아이템을 예측하나, 데이터 정보가 충분하지 않은 경우 적합한 추천을 하지 못하게 되는 이른바 cold start 문제가 발생합니다.

이러한 Cold Start Problem 중 **신규 출시되어 인터렉션이 적은 아이템들**의 Cold Start문제에 집중하여 item id embedding을 개선하는 3가지 방법론을 적용하였습니다.

:tada: [결과 ppt](https://github.com/boostcampaitech7/level4-recsys-finalproject-hackathon-recsys-09-lv3/blob/main/docs/Recsys_9%EC%A1%B0_Tving_Item-cold-start.pdf)

# Contents
💾[Dataset](#-dataset)

🧐[How to Use](#-how-to-use)

📑[Files & Parameters](#-files--parameters)

🙌[Team](#-team)


💾 Dataset
-------
### Steam Games Review Dataset
- https://www.kaggle.com/datasets/fronkongames/steam-games-dataset/data
- https://github.com/kang205/SASRec
```
# Interaction: 14,529,074
# Users:       680,812
# Items:       37,141
```

🧐 How to Use
----------
1. Steam 데이터셋을 다운받아 아래 폴더 구조처럼 압축해제
```
datahub
└─ steam      
    ├─ inter.csv
    └─ item.csv
```
2. `datahub/steam/steam_preprocessing.py` 코드 전체 실행
3. `datahub/steam/` 내에 `emb_warm_split_preprocess.pkl` 파일과 `steam_data.pkl` 파일 생성 확인
4. 터미널에서 `run.sh` 내 shell script 실행
### 실행 command
- 기본 실험 세팅 파라미터
```
python main.py --dataset_name steam --model_name deepfm  --warmup_model base --pretrain_model_path pretrain_backbones

python main.py --dataset_name steam --model_name deepfm  --warmup_model metaE  --pretrain_model_path pretrain_backbones

python main.py --dataset_name steam --model_name deepfm  --warmup_model mwuf --pretrain_model_path pretrain_backbones

python main.py --dataset_name steam --model_name deepfm  --warmup_model cvar --cvar_iters 10 --pretrain_model_path pretrain_backbones

```
- model 중 DeepFM만을 리팩토링하여 사용
- 기본 random seed는 **1234**
- 최종 결과는 `--run 10`으로 실행

실행 시 cold, warm a, warm b, warm c의 AUC / F1 Score를 출력합니다.


📑 Files & Parameters
----------
### Files
`model/*`: 다양한 backbone 모델의 구현

`model/warm.py`: 3가지 warm-up 모델의 구현

`main.py`: train, test 함수를 통해 실험 실행

`pretrain_backbones/`: backbone 모델의 pretrain 파라미터 pickle 파일이 저장되는 폴더


### Parameters

본 프로젝트에서 사용한 코드를 기준으로 한 parameter 설명입니다.

Parameter | Options | Usage
--------- | ------- | -----
--dataset_name | [steam] | Specify the dataset for evaluation
--model_name | [deepfm] | Specify the backbone for recommendation 
--warmup_model |[base, mwuf, metaE, cvar_init, cvar] | Specify the warm-up method
--is_dropoutnet | [True, False] | Specify whether to use dropoutNet for backbone pretraining
--device | [cpu, cuda:0] | Specify the device (CPU or GPU) to run the program
--runs | default 1 | Specify the number of executions to compute average metrics
--cvar_iters | default 10 | iteration count of CVAR warm up model
--pretrain_model_path | | Specify the path to store pretrained model's parameter pickle file

더 자세한 파라미터 설명 및 사용은 `./main.py` 파일을 참고하세요.


🙌 Team
------
<table>
    <tr height="140px">
        <td align="center" width="130px">	
            <a href="https://github.com/minhappy68"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/127316585?v=4"/></a>
            <br />
            <a href="https://github.com/minhappy68">minhappy68
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/imnoans"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/121077194?v=4"/></a>
            <br />
            <a href="https://github.com/imnoans">imnoans
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/eatingrabbit"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/81786179?v=4"/></a>
            <br />
            <a href="https://github.com/eatingrabbit">eatingrabbit
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/hyeonjinha"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/65064566?v=4"/></a>
            <br />
            <a href="https://github.com/hyeonjinha">hyeonjinha
        </td>
        <td align="center" width="130px">
            <a href="https://github.com/hansg931"><img height="100px" width="100px" src="https://avatars.githubusercontent.com/u/118149994?v=4"/></a>
            <br />
            <a href="https://github.com/hansg931">hansg931
        </td>
    </tr>
</table>

🔍 Citation
--------
코드 baseline: by XuZhao (<xuzzzhao@tencent.com>) [[link](https://github.com/BestActionNow/CVAR)]
```
@inproceedings{zhao2022improving,
  title={Improving Item Cold-start Recommendation via Model-agnostic Conditional Variational Autoencoder},
  author={Xu Zhao and Yi Ren and Ying Du and Shenzheng Zhang and Nian Wang},
  booktitle={SIGIR},
  year={2022},
}
```

