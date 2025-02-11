import os
import copy
from matplotlib.pyplot import axis
import torch
import random
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import argparse
from data import MovieLens1MColdStartDataLoader, TaobaoADColdStartDataLoader, SteamColdStartDataLoader
from model import FactorizationMachineModel, WideAndDeep, DeepFactorizationMachineModel, AdaptiveFactorizationNetwork, ProductNeuralNetworkModel
from model import AttentionalFactorizationMachineModel, DeepCrossNetworkModel, MWUF, MetaE, CVAR
from model.wd import WideAndDeep
from torch.utils.data import DataLoader
from typing import Optional
from typing import List, Tuple

# ================================
# 1. Argument Parsing 함수
# ================================

def get_args():
    """
    실행에 필요한 인자(argument)들을 파싱하는 함수
    """
    parser = argparse.ArgumentParser()
    
    # 데이터 및 모델 관련 설정
    parser.add_argument('--pretrain_model_path', default='/data/ephemeral/home/ksy/level4-recsys-finalproject-hackathon-recsys-09-lv3/pretrain_backbones')
    parser.add_argument('--dataset_name', default='taobaoAD', help='required to be one of [movielens1M, taobaoAD, steam]')
    parser.add_argument('--datahub_path', default='./datahub/')
    parser.add_argument('--warmup_model', default='cvar', help="required to be one of [base, mwuf, metaE, cvar, cvar_init]")
    parser.add_argument('--is_dropoutnet', type=bool, default=False, help="whether to use dropout net for pretrain")
    parser.add_argument('--model_name', default='deepfm', help='backbone name, we implemented [fm, wd, deepfm, afn, ipnn, opnn, afm, dcn]')
    
    # 학습 관련 설정
    parser.add_argument('--bsz', type=int, default=2048)
    parser.add_argument('--shuffle', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--cvar_epochs', type=int, default=2)
    parser.add_argument('--cvar_iters', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    
    # 디바이스 및 기타 설정
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument('--runs', type=int, default=1, help = 'number of executions to compute the average metrics')
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()
    return args


# ================================
# 2. 데이터 로더 생성 함수
# ================================
def get_loaders(
    name: str,
    datahub_path: str,
    device: torch.device,
    bsz: int,
    shuffle: bool
) -> SteamColdStartDataLoader:
    """
    데이터셋 이름에 따라 적절한 데이터 로더를 반환하는 함수.

    :param name: 데이터셋 이름 ('movielens1M', 'taobaoAD', 'steam' 중 하나)
    :param datahub_path: 데이터셋 파일이 위치한 기본 디렉토리 경로
    :param device: 데이터를 저장할 장치 (CPU 또는 GPU)
    :param bsz: 배치 크기
    :param shuffle: 데이터 셔플 여부 (True/False)
    
    :return: 해당 데이터셋의 DataLoader 객체
    """
    
    # 데이터 파일 경로 설정
    path = os.path.join(datahub_path, name, "{}_data.pkl".format(name))
    
    # 데이터셋별 로더 매핑
    dataset_classes = {
        "movielens1M": MovieLens1MColdStartDataLoader,
        "taobaoAD": TaobaoADColdStartDataLoader,
        "steam": SteamColdStartDataLoader
    }
    
    # 데이터셋이 지원되는 경우 해당 데이터 로더 반환
    if name in dataset_classes:
        return dataset_classes[name](name, path, device, bsz=bsz, shuffle=shuffle)
    
    return dataloaders

# ================================
# 3. 모델 생성 함수
# ================================
def get_model(name: str, dl: SteamColdStartDataLoader) :
    """
    모델 이름에 따라 적절한 모델을 반환하는 함수.

    :param name: 모델 이름 ('fm', 'wd', 'deepfm', 'afn', 'ipnn', 'opnn', 'afm', 'dcn' 중 하나)
    :param dl: 데이터셋 로더 객체 (DataLoader), 모델 초기화를 위한 description 포함
    :return: 해당 모델의 인스턴스
    """
    # 모델명과 대응하는 클래스 매핑 (객체 생성이 아닌 클래스만 저장)
    model_classes = {
        "fm": FactorizationMachineModel,
        "wd": WideAndDeep,
        "deepfm": DeepFactorizationMachineModel,
        "afn": AdaptiveFactorizationNetwork,
        "ipnn": ProductNeuralNetworkModel,
        "opnn": ProductNeuralNetworkModel,
        "afm": AttentionalFactorizationMachineModel,
        "dcn": DeepCrossNetworkModel
    }

    # 지원되지 않는 모델인 경우 예외 처리
    if name not in model_classes:
        raise ValueError(f"알 수 없는 모델 이름: {name}")

    # 해당 모델의 객체를 생성하여 반환
    if name == "ipnn" or name == "opnn":
        return model_classes[name](dl.description, embed_dim=16, mlp_dims=(16,), dropout=0, method="inner" if name == "ipnn" else "outer")

    if name == "afm":
        return model_classes[name](dl.description, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))

    if name == "dcn":
        return model_classes[name](dl.description, embed_dim=16, num_layers=3, mlp_dims=[16, 16], dropout=0.2)

    if name == "afn":
        return model_classes[name](dl.description, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400), dropout=0)

    return model_classes[name](dl.description, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)

def test(model, data_loader, device):
    """
    모델을 평가하는 함수 (AUC, F1 점수 계산).

    :param model: 평가할 PyTorch 모델
    :param data_loader: 평가할 데이터셋의 DataLoader
    :param device: 사용할 디바이스 (CPU 또는 GPU)
    
    :return: (AUC 점수, F1 점수) 튜플 반환
    """
    model.eval()
    labels, scores, predicts = list(), list(), list()
    criterion = torch.nn.BCELoss()
    
    with torch.no_grad():
        for _, (features, label) in enumerate(data_loader):
            # 데이터를 GPU/CPU에 로드
            features = {key: value.to(device) for key, value in features.items()}
            label = label.to(device)
            
            # 모델 예측값 생성
            y = model(features)
            labels.extend(label.tolist()) # 실제 정답 추가
            scores.extend(y.tolist()) # 예측 확률 추가
    
    # AUC 및 F1 점수 계산
    scores_arr = np.array(scores)
    auc = roc_auc_score(labels, scores)
    f1 = f1_score(labels, (scores_arr > np.mean(scores_arr)).astype(np.float32).tolist())
    
    return auc, f1

def dropoutNet_train(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    lr: float,
    weight_decay: float,
    save_path: str,
    log_interval: int = 10,
    val_data_loader: Optional[DataLoader] = None
) -> None:
    """
    DropoutNet을 적용하여 모델을 학습하는 함수.

    :param model: 학습할 PyTorch 모델
    :param data_loader: 학습 데이터가 포함된 DataLoader
    :param device: 사용할 디바이스 (CPU 또는 GPU)
    :param epoch: 총 학습 epoch 수
    :param lr: 학습률 (Learning Rate)
    :param weight_decay: Adam 옵티마이저의 가중치 감쇠 값
    :param save_path: 모델 저장 경로
    :param log_interval: 로그를 출력할 반복 간격 (기본값: 10)
    :param val_data_loader: 검증 데이터 로더 (기본값: None)
    
    :return: None (학습 진행)
    """
    
    # train
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # 학습 루프 시작
    for epoch_i in range(1, epoch + 1):
        epoch_loss = 0.0
        total_loss = 0
        total_iters = len(data_loader) 
        
        for i, (features, label) in enumerate(data_loader):
            bsz = label.shape[0]
            
            # 10% 확률로 아이템 임베딩을 평균값으로 대체하여 학습
            if random.random() < 0.1:
                item_emb_layer = model.emb_layer['item_id']
                origin_item_emb = item_emb_layer(features['item_id']).squeeze(1)
                mean_item_emb = torch.mean(item_emb_layer.weight.data, dim=0, keepdims=True).repeat(bsz, 1)
                y = model.forward_with_item_id_emb(mean_item_emb, features)
            else:
                y = model(features)
            
            # 손실 계산 및 역전파
            loss = criterion(y, label.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 손실 값 업데이트
            epoch_loss += loss.item()
            total_loss += loss.item()
            
            # 로그 출력
            if (i + 1) % log_interval == 0:
                avg_loss = total_loss / log_interval
                print(f"    Iter {i+1}/{total_iters} | Loss: {avg_loss:.4f}", end='\r')
                total_loss = 0 # 로그 후 초기화

        # Epoch별 손실 값 출력
        avg_epoch_loss = epoch_loss / total_iters
        print(f"Epoch {epoch_i}/{epoch} | Average Loss: {avg_epoch_loss:.4f}")
    return 

# ================================
# 5. 학습 함수
# ================================
def train(
    model: torch.nn.Module, 
    data_loader: DataLoader, 
    device: torch.device,
    epoch: int, 
    lr: float, 
    weight_decay: float, 
    save_path: str, 
    log_interval: int = 10, 
    val_data_loader: Optional[DataLoader]=None
) -> None: 
    """
    모델을 학습하는 함수
    
    :param model: 학습할 PyTorch 모델
    :param data_loader: 학습 데이터가 포함된 DataLoader
    :param device: 사용할 디바이스 (CPU 또는 GPU)
    :param epoch: 총 학습 epoch 수
    :param lr: 학습률 (Learning Rate)
    :param weight_decay: Adam 옵티마이저의 가중치 감쇠 값
    :param save_path: 모델 저장 경로
    :param log_interval: 로그를 출력할 반복 간격 (기본값: 10)
    :param val_data_loader: 검증 데이터 로더 (기본값: None)
    
    :return: None (학습 진행)
    """
    
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # 학습 루프 시작
    for epoch_i in range(1, epoch + 1):
        epoch_loss = 0.0
        total_loss = 0
        total_iters = len(data_loader) 
        
        for i, (features, label) in enumerate(data_loader):
            
            # 모델 예측
            y_pred = model(features)
            
            # 손실 계산 및 역전파
            loss = criterion(y_pred, label.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 손실 값 업데이트
            epoch_loss += loss.item()
            total_loss += loss.item()
            
            # 로그 출력
            if (i + 1) % log_interval == 0:
                avg_loss = total_loss / log_interval
                print(f"    Iter {i+1}/{total_iters} | Loss: {avg_loss:.4f}", end='\r')
                total_loss = 0 # 로그 후 초기화
        
        # Epoch 별 손실 출력
        avg_epoch_loss = epoch_loss / total_iters
        print(f"Epoch {epoch_i}/{epoch} | Average Loss: {avg_epoch_loss:.4f}")
    return 

def pretrain(
    dataset_name: str, 
    datahub_name: str,
    bsz: int,
    shuffle: bool,
    model_name: str,
    epoch: int,
    lr: float,
    weight_decay: float,
    device,
    save_dir: str,
    is_dropoutnet: bool = False
) -> Tuple[torch.nn.Module, SteamColdStartDataLoader]:
    """
    주어진 데이터셋과 모델을 사용하여 사전 학습을 수행하는 함수.

    :param dataset_name: 사용할 데이터셋 이름 (예: 'movielens1M', 'taobaoAD', 'steam')
    :param datahub_name: 데이터가 저장된 경로
    :param bsz: 배치 크기 (Batch Size)
    :param shuffle: 데이터 셔플 여부 (True/False)
    :param model_name: 학습할 모델의 이름 (예: 'deepfm', 'afn')
    :param epoch: 학습할 총 에폭 수
    :param lr: 학습률 (Learning Rate)
    :param weight_decay: Adam 옵티마이저의 가중치 감쇠 값
    :param device: 사용할 장치 (예: 'cuda:0', 'cpu')
    :param save_dir: 학습된 모델을 저장할 경로
    :param is_dropoutnet: DropoutNet 적용 여부 (True/False)

    :return: 학습된 모델과 데이터 로더 (model, data_loaders)
    """
    
    device = torch.device(device)
    save_dir = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리 생성 (존재하면 건너뜀)
    
    data_loaders = get_loaders(dataset_name, datahub_name, device, bsz, shuffle==True)
    model = get_model(model_name, data_loaders).to(device)
    save_path = os.path.join(save_dir, 'model.pth')
    
    print("=" * 20, f"Pretraining {model_name}", "=" * 20)
    
    model.init()
    
    # DropoutNet 적용 여부에 따라 학습 방식 선택
    if is_dropoutnet:
        dropoutNet_train(
            model, data_loaders['train_base'], device, epoch, lr, weight_decay, save_path, 
            val_data_loader=data_loaders['test']
        )
    else:
        train(model, data_loaders['train_base'], device, epoch, lr, weight_decay, save_path,
              val_data_loader=data_loaders['test']
        )
    
    print("=" * 20, f"Pretraining {model_name} Completed", "=" * 20)
    
    return model, data_loaders

def base(
    model: torch.nn.Module,
    dataloaders: dict,
    model_name: str,
    epoch: int,
    lr: float,
    weight_decay: float,
    device: str,
    save_dir: str
) -> Tuple[List[float], List[float]]:
    """
    기본 모델 학습 및 평가를 수행하는 함수.

    :param model: 학습할 PyTorch 모델
    :param dataloaders: 데이터 로더 딕셔너리 (train/test 포함)
    :param model_name: 학습할 모델의 이름
    :param epoch: 총 학습 epoch 수
    :param lr: 학습률 (Learning Rate)
    :param weight_decay: Adam 옵티마이저의 가중치 감쇠 값
    :param device: 사용할 장치 (예: 'cuda:0', 'cpu')
    :param save_dir: 모델이 저장될 디렉토리

    :return: AUC 및 F1 점수 리스트 (auc_list, f1_list)
    """
    
    print("*" * 20, "Base Training", "*" * 20)
    
    device = torch.device(device)
    
    # 모델 저장 디렉토리 설정
    save_dir = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리 생성 (존재하면 건너뜀)

    # 모델 저장 경로
    save_path = os.path.join(save_dir, "model.pth")
    
    # 평가할 데이터셋 목록
    dataset_list = ["train_warm_a", "train_warm_b", "train_warm_c", "test"]
    
    # AUC 및 F1 점수를 저장할 리스트
    auc_list, f1_list = [], []

    # 각 데이터셋에 대해 평가 및 학습 수행
    for i, dataset_name in enumerate(dataset_list):
        # 모델 평가 수행 (테스트 데이터셋 사용)
        auc, f1 = test(model, dataloaders['test'], device)
        auc_list.append(auc)
        f1_list.append(f1)
        
        print(f"[Base Model] Evaluating on [Test Dataset] | AUC: {auc:.4f}, F1 Score: {f1:.4f}")
        
        # holdout 제외하고 학습 진행
        if i < len(dataset_list) - 1:
            model.only_optimize_itemid() # 아이템 ID 최적화 수행
            train(model, dataloaders[dataset_name], device, epoch, lr, weight_decay, save_path)
    
    print("*" * 20, "Base Training Completed", "*" * 20)

    return auc_list, f1_list

def metaE(
    model,
    dataloaders,
    model_name: str,
    epoch: int,
    lr: float,
    weight_decay: float,
    device: str,
    save_dir:str
) -> Tuple[List[float], List[float]]:
    """
    Meta Embedding (MetaE) 학습 및 평가를 수행하는 함수.

    :param model: 학습할 PyTorch 모델
    :param dataloaders: 데이터 로더 딕셔너리
    :param model_name: 학습할 모델의 이름
    :param epoch: 총 학습 epoch 수
    :param lr: 학습률 (Learning Rate)
    :param weight_decay: Adam 옵티마이저의 가중치 감쇠 값
    :param device: 사용할 장치 (예: 'cuda:0', 'cpu')
    :param save_dir: 모델 저장 디렉토리 경로

    :return: AUC 및 F1 점수 리스트 (auc_list, f1_list)
    """
    
    print("*" * 20, "MetaE Training", "*" * 20)
    
    device = torch.device(device)
    
    # 모델 저장 디렉토리 설정
    save_dir = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리 생성 (존재하면 건너뜀)

    # 모델 저장 경로
    save_path = os.path.join(save_dir, "model.pth")
    
    train_base = dataloaders['train_base']
    metaE_model = MetaE(model, warm_features=dataloaders.item_features, device=device).to(device)
    
    # MetaE 데이터 로더 가져오기
    metaE_dataloaders = [dataloaders[name] for name in ['metaE_a', 'metaE_b', 'metaE_c', 'metaE_d']]
    
    # MetaE 학습 설정
    metaE_model.train()
    criterion = torch.nn.BCELoss()
    metaE_model.optimize_metaE()
    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, metaE_model.parameters()),
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # MetaE 학습
    for epoch_i in range(epoch):
        dataloader_a = metaE_dataloaders[epoch_i]
        dataloader_b = metaE_dataloaders[(epoch_i + 1) % 4]
        
        epoch_loss = 0.0
        total_iter_num = len(dataloader_a)
        iter_dataloader_b = iter(dataloader_b)
        
        for i, (features_a, label_a) in enumerate(dataloader_a):
            features_b, label_b = next(iter_dataloader_b)
            
            # Meta E 손실 계산
            loss_a, target_b = metaE_model(features_a, label_a, features_b, criterion)
            loss_b = criterion(target_b, label_b.float())
            loss = 0.1 * loss_a + 0.9 * loss_b
            
            # 역전파 및 최적화
            metaE_model.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"    Iter {i+1}/{total_iter_num} | Loss: {loss:.4f}, Loss A: {loss_a:.4f}, Loss B: {loss_b:.4f}", end='\r')
        
        avg_epoch_loss = epoch_loss / total_iter_num
        print(f"Epoch {epoch_i+1}/{epoch} | Average Loss: {avg_epoch_loss:.4f}")
    
    # 기존 아이템 ID 임베딩을 MetaE로 업데이트
    train_a = dataloaders['train_warm_a']
    for (features, label) in train_a:
        origin_item_id_emb = metaE_model.model.emb_layer[metaE_model.item_id_name].weight.data
        warm_item_id_emb = metaE_model.warm_item_id(features)
        indexes = features[metaE_model.item_id_name].squeeze()
        origin_item_id_emb[indexes, ] = warm_item_id_emb
    
    # test by steps 
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    auc_list, f1_list = [], []
    
    # 학습 후 평가가
    for i, train_s in enumerate(dataset_list):
        print("#" * 10, train_s, "#" * 10)
        
        train_s = dataset_list[i]
        
        auc, f1 = test(metaE_model.model, dataloaders['test'], device)
        auc_list.append(auc)
        f1_list.append(f1)
        
        print(f"[MetaE] Evaluating on [Test Dataset] | AUC: {auc:.4f}, F1 Score: {f1:.4f}")
        
        if i < len(dataset_list) - 1:
            metaE_model.model.only_optimize_itemid()
            train(metaE_model.model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
    
    print("*" * 20, "MetaE Training Completed", "*" * 20)
    return auc_list, f1_list

def mwuf(
    model: torch.nn.Module,
    dataloaders,
    model_name: str,
    epoch: int,
    lr: float,
    weight_decay: float,
    device: str,
    save_dir: str
) -> Tuple[List[float], List[float]]:
    """
    MWUF (Meta-Weight Update Framework) 학습 및 평가를 수행하는 함수.

    :param model: 학습할 PyTorch 모델
    :param dataloaders: 데이터 로더 딕셔너리
    :param model_name: 학습할 모델의 이름
    :param epoch: 총 학습 epoch 수
    :param lr: 학습률 (Learning Rate)
    :param weight_decay: Adam 옵티마이저의 가중치 감쇠 값
    :param device: 사용할 장치 (예: 'cuda:0', 'cpu')
    :param save_dir: 모델 저장 디렉토리 경로

    :return: AUC 및 F1 점수 리스트 (auc_list, f1_list)
    """
    
    print("*" * 20, "MWUF Training", "*" * 20)
    
    device = torch.device(device)
    
    # 모델 저장 디렉토리 설정
    save_dir = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리 생성 (존재하면 건너뜀)

    # 모델 저장 경로
    save_path = os.path.join(save_dir, "model.pth")
    
    train_base = dataloaders['train_base']
    
    # train mwuf
    mwuf_model = MWUF(model, 
                      item_features=dataloaders.item_features,
                      train_loader=train_base,
                      device=device).to(device)
    
    mwuf_model.init_meta()
    mwuf_model.train()
    
    # 손실 함수 및 최적화 설정
    criterion = torch.nn.BCELoss()
    mwuf_model.optimize_new_item_emb()
    optimizer1 = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, mwuf_model.parameters()),
        lr=lr, 
        weight_decay=weight_decay
    )
    
    mwuf_model.optimize_meta()
    optimizer2 = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, 
        mwuf_model.parameters()), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    mwuf_model.optimize_all()
    
    # MWUF 학습 루프
    total_iters = len(train_base)
    loss_1, loss_2 = 0.0, 0.0
    
    for i, (features, label) in enumerate(train_base):
        # if i + 1 > total_iters * 0.3:
        #     break
        y_cold = mwuf_model.cold_forward(features)
        cold_loss = criterion(y_cold, label.float())
        
        mwuf_model.zero_grad()
        cold_loss.backward()
        optimizer1.step()
        
        y_warm = mwuf_model.forward(features)
        warm_loss = criterion(y_warm, label.float())
        
        mwuf_model.zero_grad()
        warm_loss.backward()
        optimizer2.step()
        
        loss_1 += cold_loss
        loss_2 += warm_loss
        
        if (i + 1) % 10 == 0:
            print(f"    Iter {i+1}/{total_iters} | Warm Loss: {warm_loss.item():.4f}", end='\r')
    
    # 최종 평균 손실 출력
    avg_cold_loss = loss_1 / total_iters
    avg_warm_loss = loss_2 / total_iters
    print(f"Final Average Warm-up Loss: Cold Loss: {avg_cold_loss:.4f}, Warm Loss: {avg_warm_loss:.4f}")

    # 훈련된 메타 임베딩을 새로운 아이템 임베딩 초기화에 활용
    train_a = dataloaders['train_warm_a']
    for (features, label) in train_a:
        origin_item_id_emb = mwuf_model.model.emb_layer[mwuf_model.item_id_name].weight.data
        warm_item_id_emb = mwuf_model.warm_item_id(features)
        indexes = features[mwuf_model.item_id_name].squeeze()
        origin_item_id_emb[indexes, ] = warm_item_id_emb
    
    # 평가 데이터셋 리스트
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    auc_list, f1_list = [], []
    
    for i, train_s in enumerate(dataset_list):
        print("#" * 10, dataset_list[i], '#' * 10)
        train_s = dataset_list[i]
        
        # 모델 평가
        auc, f1 = test(mwuf_model.model, dataloaders['test'], device)
        auc_list.append(auc)
        f1_list.append(f1)
        
        print(f"[MWUF] Evaluating on [Test Dataset] | AUC: {auc:.4f}, F1 Score: {f1:.4f}")
        
        # 학습 수행
        if i < len(dataset_list) - 1:
            mwuf_model.model.only_optimize_itemid()
            train(mwuf_model.model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
    
    print("*" * 20, "MWUF Training Completed", "*" * 20)
    
    return auc_list, f1_list

def cvar(
    model: torch.nn.Module,
    dataloaders,
    model_name: str,
    epoch: int,
    cvar_epochs: int,
    cvar_iters: int,
    lr: float,
    weight_decay: float,
    device: str,
    save_dir: str,
    only_init: bool=False
) -> Tuple[List[float], List[float]]:
    """
    CVAR (Conditional Variance Adjustment Representation) 학습 및 평가를 수행하는 함수.

    :param model: 학습할 PyTorch 모델
    :param dataloaders: 데이터 로더 딕셔너리
    :param model_name: 학습할 모델의 이름
    :param epoch: 총 학습 epoch 수
    :param cvar_epochs: CVAR 학습 epoch 수
    :param cvar_iters: CVAR 학습 반복 횟수
    :param lr: 학습률 (Learning Rate)
    :param weight_decay: Adam 옵티마이저의 가중치 감쇠 값
    :param device: 사용할 장치 (예: 'cuda:0', 'cpu')
    :param save_dir: 모델 저장 디렉토리 경로
    :param only_init: 초기화 후 추가 학습을 진행할지 여부

    :return: AUC 및 F1 점수 리스트 (auc_list, f1_list)
    """
    
    print("*" * 20, "CVAR Training", "*" * 20)
    
    device = torch.device(device)
    
    # 모델 저장 디렉토리 설정
    save_dir = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리 생성 (존재하면 건너뜀)

    # 모델 저장 경로
    save_path = os.path.join(save_dir, "model.pth")

    train_base = dataloaders['train_base']
    
    # cvar 모델 초기화
    warm_model = CVAR(
        model, 
        warm_features=dataloaders.item_features,
        train_loader=train_base,
        device=device).to(device)
    
    warm_model.init_cvar()
    
    def warm_up(dataloader, epochs, iters, logger=False):
        """
        CVAR 모델을 사전 학습하는 함수.

        :param dataloader: 학습에 사용할 DataLoader
        :param epochs: 학습 epoch 수
        :param iters: 반복 학습 횟수
        :param logger: 로그 출력 여부
        """
        warm_model.train()
        criterion = torch.nn.BCELoss()
        warm_model.optimize_cvar()
        optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, warm_model.parameters()),
            lr=lr, 
            weight_decay=weight_decay
        )
        
        batch_num = len(dataloader)
        
        # train warm-up model
        for e in range(epochs):
            for i, (features, label) in enumerate(dataloader):
                total_loss, total_main_loss, total_recon_loss, total_reg_loss = 0.0, 0.0, 0.0, 0.0
                
                for _ in range(iters):
                    target, recon_term, reg_term  = warm_model(features)
                    main_loss = criterion(target, label.float())
                    loss = main_loss + recon_term + 1e-4 * reg_term
                    
                    warm_model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_main_loss += main_loss.item()
                    total_recon_loss += recon_term.item()
                    total_reg_loss += reg_term.item()
                    
                avg_loss = total_loss / iters
                avg_main_loss = total_main_loss / iters
                avg_recon_loss = total_recon_loss / iters
                avg_reg_loss = total_reg_loss / iters
                
                if logger and (i + 1) % 10 == 0:
                    print(f"    Iter {i+1}/{batch_num} | Loss: {avg_loss:.4f}, Main: {avg_main_loss:.4f}, Recon: {avg_recon_loss:.4f}, Reg: {avg_reg_loss:.4f}", end='\r')
        
        # warm-up 아이템 ID 임베딩 업데이트
        train_a = dataloaders['train_warm_a']
        for features, label in train_a:
            origin_item_id_emb = warm_model.model.emb_layer[warm_model.item_id_name].weight.data
            warm_item_id_emb, _, _ = warm_model.warm_item_id(features)
            indexes = features[warm_model.item_id_name].squeeze()
            origin_item_id_emb[indexes, ] = warm_item_id_emb
    
    # CVAR을 old item에 대하여 학습 수행
    warm_up(train_base, epochs=1, iters=cvar_iters, logger=True)
    
    # 평가 데이터셋 리스트
    dataset_list = ['train_warm_a', 'train_warm_b', 'train_warm_c', 'test']
    auc_list, f1_list = [], []
    
    # 평가 루프
    for i, train_s in enumerate(dataset_list):
        print("#"*10, dataset_list[i],'#'*10)
        train_s = dataset_list[i]
        
        # 모델 평가
        auc, f1 = test(warm_model.model, dataloaders['test'], device)
        auc_list.append(auc)
        f1_list.append(f1)
        
        print(f"[CVAR] Evaluating on [Test Dataset] | AUC: {auc:.4f}, F1 Score: {f1:.4f}")

        # 마지막 테스트 단계 이전까지 학습 수행
        if i < len(dataset_list) - 1:
            warm_model.model.only_optimize_itemid()
            train(warm_model.model, dataloaders[train_s], device, epoch, lr, weight_decay, save_path)
            
            # only_init가 아닌 경우 warm_up 수행
            if not only_init:
                warm_up(dataloaders[train_s], epochs=cvar_epochs, iters=cvar_iters, logger=False)
    
    print("*" * 20, "CVAR Training Completed", "*" * 20)
    
    return auc_list, f1_list

def run(
    model: torch.nn.Module,
    dataloaders,
    args,
    model_name: str,
    warm: str
) -> Tuple[list, list]:
    """
    주어진 모델과 데이터 로더를 사용하여 다양한 사전 학습 기법을 실행하는 함수.

    :param model: 학습할 PyTorch 모델
    :param dataloaders: 데이터 로더 딕셔너리
    :param args: 하이퍼파라미터 및 설정 값이 포함된 객체
    :param model_name: 학습할 모델의 이름
    :param warm: 사용할 사전 학습 기법 ('base', 'mwuf', 'metaE', 'cvar', 'cvar_init' 중 하나)

    :return: AUC 및 F1 점수 리스트 (auc_list, f1_list)
    """
    if warm == 'base':
        auc_list, f1_list = base(model, dataloaders, model_name, args.epoch, args.lr, args.weight_decay, args.device, args.save_dir)
    elif warm == 'mwuf':
        auc_list, f1_list = mwuf(model, dataloaders, model_name, args.epoch, args.lr, args.weight_decay, args.device, args.save_dir)
    elif warm == 'metaE': 
        auc_list, f1_list = metaE(model, dataloaders, model_name, args.epoch, args.lr, args.weight_decay, args.device, args.save_dir)
    elif warm == 'cvar': 
        auc_list, f1_list = cvar(model, dataloaders, model_name, args.epoch, args.cvar_epochs, args.cvar_iters, args.lr, args.weight_decay, args.device, args.save_dir)
    elif warm == 'cvar_init': 
        auc_list, f1_list = cvar(model, dataloaders, model_name, args.epoch, args.cvar_epochs, args.cvar_iters, args.lr, args.weight_decay, args.device, args.save_dir, only_init=True)
    
    return auc_list, f1_list

# ================================
# 6. 메인 실행 로직
# ================================
if __name__ == '__main__':
    args = get_args()
    
    if args.seed > -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    res = {}
    print(f"Model: {args.model_name}")
    torch.cuda.empty_cache()
    
    # 사전 학습 모델 로드 또는 학습
    drop_suffix = '-dropoutnet' if args.is_dropoutnet else ''
    model_filename = f"{args.model_name}{drop_suffix}-{args.dataset_name}-{args.seed}"
    model_path = os.path.join(args.pretrain_model_path, model_filename)
    
    # ================================
    # 사전 학습된 모델 로드 또는 학습 진행
    # ================================
    
    if os.path.exists(model_path):
        # 기존 모델 로드
        model = torch.load(model_path).to(args.device)
        dataloaders = get_loaders(args.dataset_name, args.datahub_path, args.device, args.bsz, args.shuffle==1)
        print(f"Loaded pre-trained model from {model_path}")
    else:
        # 모델 학습 및 저장
        model, dataloaders = pretrain(
            args.dataset_name, args.datahub_path, args.bsz, args.shuffle, args.model_name,
            args.epoch, args.lr, args.weight_decay, args.device, args.save_dir, args.is_dropoutnet
        )
        
        if len(args.pretrain_model_path) > 0:
            if not os.path.exists(args.pretrain_model_path):
                os.makedirs(args.pretrain_model_path)
            torch.save(model, model_path)
            print(f"Saved pre-trained model to {model_path}")
    
    # ================================
    # Warm-up 학습 및 테스트 실행
    # ================================
    avg_auc_list, avg_f1_list = [], []
    for i in range(args.runs):
        model_v = copy.deepcopy(model).to(args.device)
        auc_list, f1_list = run(model_v, dataloaders, args, args.model_name, args.warmup_model)
        avg_auc_list.append(np.array(auc_list))
        avg_f1_list.append(np.array(f1_list))
    
    # 결과 평균 계산
    avg_auc_list = list(np.stack(avg_auc_list).mean(axis=0))
    avg_f1_list = list(np.stack(avg_f1_list).mean(axis=0))
    
    print(f"AUC Scores: {avg_auc_list}")
    print(f"F1 Scores: {avg_f1_list}")