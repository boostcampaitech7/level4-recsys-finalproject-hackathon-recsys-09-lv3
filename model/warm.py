import torch
import torch.nn as nn
import torch.autograd as autograd
import os
import pickle as pkl

class DropoutNet(nn.Module):
    """
    DropoutNet 모델 클래스.

    특정 아이템 ID 정보를 제거한 상태에서도 학습할 수 있도록 설계된 네트워크.

    :param model: 원본 모델 (기본 추천 모델)
    :param device: 모델이 학습될 디바이스 ('cuda' 또는 'cpu')
    :param item_id_name: 아이템 ID 필드 이름 (기본값: 'item_id')
    """

    def __init__(self, model: nn.Module, device, item_id_name='item_id'):
        super(DropoutNet, self).__init__()
        self.model = model
        self.item_id_name = item_id_name
        self.mean_item_emb = self._compute_mean_item_embedding()
        return
    
    def _compute_mean_item_embedding(self) -> torch.Tensor:
        """ 아이템 임베딩의 평균값을 계산하여 DropoutNet용 임베딩 생성 """
        item_emb = self.model.emb_layer[self.item_id_name]
        return torch.mean(item_emb.weight.data, dim=0, keepdims=True).repeat(item_emb.num_embeddings, 1)

    def forward_without_itemid(self, xdict: dict) -> torch.Tensor:
        """
        아이템 ID를 사용하지 않고 모델을 실행하는 함수.

        :param xdict: 입력 데이터 딕셔너리
        :return: 모델 예측값 (Tensor)
        """
        bsz = xdict[self.item_id_name].shape[0]
        mean_emb = self.mean_item_emb.repeat(bsz, 1)
        target = self.model.forward_with_item_id_emb(mean_emb, xdict)
        return target

    def forward(self, xdict: dict) -> torch.Tensor:
        """
        일반적인 forward 함수. 아이템 ID 임베딩을 사용하여 모델 실행.

        :param xdict: 입력 데이터 딕셔너리
        :return: 모델 예측값 (Tensor)
        """
        item_id_emb = xdict[self.item_id_name]
        target = self.model.forward_with_item_id_emb(item_id_emb, xdict)
        return target

class MetaE(nn.Module):
    """
    Meta Embedding (MetaE) 모델 클래스.

    아이템 관련 특징을 활용하여 새로운 아이템 ID 임베딩을 학습하는 메타 러닝 모델.

    :param model: 원본 추천 모델
    :param warm_features: 아이템 관련 특징 목록
    :param device: 모델이 학습될 디바이스 ('cuda' 또는 'cpu')
    :param item_id_name: 아이템 ID 필드 이름 (기본값: 'item_id')
    :param emb_dim: 임베딩 차원 (기본값: 16)
    """
    def __init__(
        self, 
        model: nn.Module,
        warm_features: list,
        device,
        item_id_name: str = 'item_id',
        emb_dim: int = 16
    ):
        super(MetaE, self).__init__()
        
        self.build(model, warm_features, device, item_id_name, emb_dim)
        return 

    def build(
        self,
        model: nn.Module,
        item_features: list,
        device,
        item_id_name: str = 'item_id',
        emb_dim: int = 16
    ):
        self.model = model
        self.device = device
        self.item_id_name = item_id_name
        self.item_features = []
        
        assert item_id_name in model.item_id_name, f"Illegal item ID name: {item_id_name}"
        
        output_embedding_size = 0
        for item_f in item_features:
            assert item_f in model.features, f"Unknown feature: {item_f}"
            type = self.model.description[item_f][1]
            
            if type == 'spr' or type == 'seq':
                output_embedding_size += emb_dim
            elif type == 'ctn':
                output_embedding_size += 1
            elif type == 'emb':
                # Add the embedding dimension from the feature's description
                output_embedding_size += self.model.description[item_f][0]

            else:
                raise ValueError(f"Illegal feature type for warm: {item_f}")
            
            
            self.item_features.append(item_f) 

        # 아이템 ID를 생성하는 메타 모델
        self.itemid_generator = nn.Sequential(
            nn.Linear(output_embedding_size, 16),
            nn.ReLU(),
            nn.Linear(16, emb_dim),
        )
        
        return

    def init_metaE(self):
        """MetaE 모델의 가중치를 초기화"""
        for name, param in self.named_parameters():
            if 'itemid_generator' in name:
                torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_metaE(self):
        """MetaE 모델에서 itemid_generator만 학습 가능하도록 설정"""
        for name, param in self.named_parameters():
            if 'itemid_generator' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return

    def warm_item_id(self, x_dict):
        """
        입력된 아이템 특징을 활용하여 새로운 아이템 ID 임베딩을 생성.

        :param x_dict: 입력 데이터 딕셔너리
        :return: 예측된 아이템 ID 임베딩 (Tensor)
        """
        # get embedding of item features
        item_embs = []
        for item_f in self.item_features: 
            type = self.model.description[item_f][1]
            x = x_dict[item_f]
            
            if type == 'spr':
                emb = self.model.emb_layer[item_f](x).squeeze()
            elif type == 'ctn':
                emb = x
            elif type == 'seq':
                emb = self.model.emb_layer[item_f](x).sum(dim=1, keepdim=True).squeeze()
            elif type == 'emb':
                emb = x  # Use precomputed embedding directly
            else:
                raise ValueError(f"Illegal feature type for warm: {item_f}")
            
            item_embs.append(emb)
        
        # sideinfo_emb = torch.mean(torch.stack(item_embs, dim=1), dim=1)
        # 특징 임베딩을 연결하여 최종 아이템 ID 생성
        sideinfo_emb = torch.concat(item_embs, dim=1)
        pred = self.itemid_generator(sideinfo_emb)
        return pred

    def forward(
        self, 
        features_a, 
        label_a, 
        features_b, 
        criterion=torch.nn.BCELoss(), 
        lr:float=0.001
    ):
        """
        MetaE 모델의 Forward 과정.

        :param features_a: 첫 번째 데이터 샘플 (새로운 아이템 ID를 학습하는 입력)
        :param label_a: 첫 번째 샘플의 실제 라벨
        :param features_b: 두 번째 데이터 샘플 (새로운 아이템 ID를 적용하여 평가)
        :param criterion: 손실 함수 (기본값: BCELoss)
        :param lr: 학습률 (기본값: 0.001)

        :return: (첫 번째 샘플의 손실값, 두 번째 샘플의 예측값)
        """
        # 새로운 아이템 ID 임베딩 생성 
        new_item_id_emb = self.warm_item_id(features_a)
        
        # batch_a를 사용하여 손실 계산
        target_a = self.model.forward_with_item_id_emb(new_item_id_emb, features_a)
        loss_a = criterion(target_a, label_a.float())
        
        # 손실에 대한 그래디언트 계산 및 업데이트
        grad = autograd.grad(loss_a, new_item_id_emb, create_graph=True)[0]
        new_item_id_emb_update = new_item_id_emb - lr * grad
        
        # batch_b를를 사용하여 평가
        target_b = self.model.forward_with_item_id_emb(new_item_id_emb_update, features_b)
        return loss_a, target_b

class MWUF(nn.Module):
    """
    Meta Weight Update Framework (MWUF) 모델.

    아이템 특징을 활용하여 새로운 아이템 ID 임베딩을 학습하고 사용자 정보를 반영하는 메타 러닝 모델.

    :param model: 원본 추천 모델
    :param item_features: 아이템 관련 특징 목록
    :param train_loader: 학습 데이터 로더
    :param device: 모델이 학습될 디바이스 ('cuda' 또는 'cpu')
    :param item_id_name: 아이템 ID 필드 이름 (기본값: 'item_id')
    :param emb_dim: 임베딩 차원 (기본값: 16)
    """
    def __init__(
        self, 
        model: nn.Module,
        item_features: list,
        train_loader,
        device,
        item_id_name: str = 'item_id',
        emb_dim: int = 16):
        super(MWUF, self).__init__()
        self.build(model, item_features, train_loader, device, item_id_name, emb_dim)
        return 

    def build(
        self,
        model: nn.Module,
        item_features: list,
        train_loader,
        device,
        item_id_name: str = 'item_id',
        emb_dim = 16):

        self.model = model
        self.device = device
        self.item_id_name = item_id_name
        self.item_features = []
        self.output_emb_size = 0
        
        assert item_id_name in model.item_id_name, f"Illegal item ID name: {item_id_name}"
        
        for item_f in item_features:
            assert item_f in model.features, f"Unknown feature: {item_f}"
            type = self.model.description[item_f][1]
            
            if type == 'spr' or type == 'seq':
                self.output_emb_size += emb_dim
            elif type == 'ctn':
                self.output_emb_size += 1
            elif type == 'emb':
                # Add the actual embedding dimension
                self.output_emb_size += self.model.description[item_f][0]
            else:
                raise ValueError('illegal feature type for warm: {}'.format(item_f))
            
            self.item_features.append(item_f)
        
        # 새로운 아이템 임베딩 생성
        item_emb = self.model.emb_layer[self.item_id_name]
        new_item_emb = torch.mean(item_emb.weight.data, dim=0, keepdims=True).repeat(item_emb.num_embeddings, 1)
        self.new_item_emb = nn.Embedding.from_pretrained(new_item_emb, freeze=False)
        # self.new_item_emb = nn.Embedding(item_emb.num_embeddings, item_emb.embedding_dim)
        
        # Meta shift 및 scale 네트워크
        self.meta_shift = nn.Sequential(
            nn.Linear(emb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, emb_dim)
        )
        self.meta_scale = nn.Sequential(
            nn.Linear(self.output_emb_size, 16),
            nn.ReLU(),
            nn.Linear(16, emb_dim)
        )
        
        # 아이템-사용자 평균 임베딩 로딩
        self.get_item_avg_users_emb(train_loader, device)
        return

    def init_meta(self):
        """Meta 네트워크 가중치 초기화"""
        for name, param in self.named_parameters():
            if ('meta_scale') in name or ('meta_shift' in name):
                torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_meta(self):
        """Meta 네트워크만 학습 가능하도록 설정"""
        for name, param in self.named_parameters():
            if ('meta_shift' in name) or ('meta_scale' in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return
    
    def optimize_new_item_emb(self):
        """새로운 아이템 임베딩만 학습 가능하도록 설정"""
        for name, param in self.named_parameters():
            if 'new_item_emb' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def optimize_all(self):
        """모든 네트워크 학습 가능하도록 설정"""
        for name, param in self.named_parameters():
            param.requires_grad_(True)
        return

    def cold_forward(self, x_dict):
        """
        Cold-start 상황에서 새로운 아이템 ID 임베딩을 사용하여 예측 수행.

        :param x_dict: 입력 데이터 딕셔너리
        :return: 예측값 (Tensor)
        """
        item_id_emb = self.new_item_emb(x_dict[self.item_id_name])
        target = self.model.forward_with_item_id_emb(item_id_emb, x_dict)
        return target

    def warm_item_id(self, x_dict):
        """
        아이템 특징 및 사용자 정보를 활용하여 새로운 아이템 ID 임베딩 생성.

        :param x_dict: 입력 데이터 딕셔너리
        :return: 새롭게 생성된 아이템 ID 임베딩 (Tensor)
        """
        item_ids = x_dict[self.item_id_name]
        item_id_emb = self.new_item_emb(item_ids).detach().squeeze()
        user_emb = self.avg_users_emb(item_ids).detach().squeeze()
        
        if user_emb.sum() == 0:
            user_emb = self.model.emb_layer['user_id'](x_dict['user_id']).squeeze()
        
        item_embs = []
        for item_f in self.item_features: 
            type = self.model.description[item_f][1]
            x = x_dict[item_f]
            if type == 'spr':
                emb = self.model.emb_layer[item_f](x).squeeze()
            elif type == 'ctn':
                emb = x
            elif type == 'seq':
                emb = self.model.emb_layer[item_f](x).sum(dim=1, keepdim=True).squeeze()
            elif type == 'emb':
                emb = x  # Use precomputed embedding directly
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            
            item_embs.append(emb)
        item_emb = torch.concat(item_embs, dim=1).detach()
        
        # scaling & shifting
        scale = self.meta_scale(item_emb)
        shift = self.meta_shift(user_emb)
        warm_item_id_emb = (scale * item_id_emb + shift)
        return warm_item_id_emb

    def forward(self, x_dict):
        """
        Warm-start 상황에서 새로운 아이템 ID 임베딩을 활용하여 예측 수행.

        :param x_dict: 입력 데이터 딕셔너리
        :return: 예측값 (Tensor)
        """
        warm_item_id_emb = self.warm_item_id(x_dict).unsqueeze(1)
        target = self.model.forward_with_item_id_emb(warm_item_id_emb, x_dict)
        return target

    def get_item_avg_users_emb(self, data_loaders, device):
        """
        아이템 별 평균 사용자 임베딩을 계산하여 저장.

        :param data_loader: 학습 데이터 로더
        """
        dataset_name = data_loaders.dataset.dataset_name
        path = f"./datahub/item2users/{dataset_name}_item2users.pkl"
        if os.path.exists(path):
            with open(path, 'rb+') as f:
                item2users = pkl.load(f)
        else:
            item2users = {}
            for features, _ in data_loaders:
                u_ids = features['user_id'].squeeze().tolist()
                i_ids = features['item_id'].squeeze().tolist()
                for i in range(len(i_ids)):
                    iid, uid = u_ids[i], i_ids[i]
                    if iid not in item2users.keys():
                        item2users[iid] = []
                    item2users[iid].append(uid)
            
            with open(path, 'wb+') as f:
                pkl.dump(item2users, f)
        
        avg_users_emb = []
        emb_dim = self.model.emb_layer[self.item_id_name].embedding_dim
        
        for item in range(self.model.emb_layer[self.item_id_name].num_embeddings):
            if item in item2users.keys():
                users = torch.Tensor(item2users[item]).long().to(device)
                avg_users_emb.append(self.model.emb_layer['user_id'](users).mean(dim=0))
            else:
                avg_users_emb.append(torch.zeros(emb_dim).to(device))
        
        avg_users_emb = torch.stack(avg_users_emb, dim=0) 
        self.avg_users_emb = nn.Embedding.from_pretrained(avg_users_emb, freeze=True)
        return

class CVAR(nn.Module):
    """
    Conditional Variance Adjustment Representation (CVAR) 모델.

    아이템의 특징을 기반으로 새로운 아이템 ID 임베딩을 생성하는 메타 러닝 모델.

    :param model: 원본 추천 모델
    :param warm_features: 아이템 관련 특징 목록
    :param device: 모델이 학습될 디바이스 ('cuda' 또는 'cpu')
    :param item_id_name: 아이템 ID 필드 이름 (기본값: 'item_id')
    :param emb_dim: 임베딩 차원 (기본값: 16)
    """
    def __init__(
        self, 
        model: nn.Module,
        warm_features: list,
        train_loader,
        device,
        item_id_name: str = 'item_id',
        emb_dim: int = 16):
        super(CVAR, self).__init__()
        self.build(model, warm_features, train_loader, device, item_id_name, emb_dim)
        return 

    def build(
        self,
        model: nn.Module,
        item_features: list,
        train_loader,
        device,
        item_id_name: str = 'item_id',
        emb_dim: int = 16):
        
        self.model = model
        self.device = device
        self.item_id_name = item_id_name
        self.item_features = []
        self.output_emb_size = 0
        self.warmup_emb_layer = nn.ModuleDict()
        
        assert item_id_name in model.item_id_name, f"Illegal item ID name: {item_id_name}"
        
        for item_f in item_features:
            assert item_f in model.features, f"Unknown feature: {item_f}"
            size, type = self.model.description[item_f]
            
            if type == 'spr' or type == 'seq':
                self.output_emb_size += emb_dim
                self.warmup_emb_layer["warmup_{}".format(item_f)] = nn.Embedding(size, emb_dim)
            elif type == 'ctn':
                self.output_emb_size += 1
            elif type == 'emb':
                self.output_emb_size += size
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            
            self.item_features.append(item_f) 
        
        self.origin_item_emb = self.model.emb_layer[self.item_id_name]
        
        # Variational encoder 및 decoder
        self.mean_encoder = nn.Linear(emb_dim, 16)
        self.log_v_encoder = nn.Linear(emb_dim, 16)
        self.mean_encoder_p = nn.Linear(self.output_emb_size, 16)
        self.log_v_encoder_p = nn.Linear(self.output_emb_size, 16)
        self.decoder = nn.Linear(17, 16)
        return

    def wasserstein(self, mean1: torch.Tensor, log_v1: torch.Tensor, mean2: torch.Tensor, log_v2: torch.Tensor) -> torch.Tensor:
        """
        Wasserstein 거리 계산.

        :param mean1: 첫 번째 평균 벡터
        :param log_v1: 첫 번째 로그 분산 벡터
        :param mean2: 두 번째 평균 벡터
        :param log_v2: 두 번째 로그 분산 벡터
        :return: Wasserstein 거리 (Tensor)
        """
        p1 = torch.sum(torch.pow(mean1 - mean2, 2), 1)
        p2 = torch.sum(torch.pow(torch.sqrt(torch.exp(log_v1)) - torch.sqrt(torch.exp(log_v2)), 2), 1)
        return torch.sum(p1 + p2)

    def init_all(self) -> None:
        """모든 네트워크 가중치를 초기화"""
        for name, param in self.named_parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_all(self) -> None:
        """모든 네트워크 학습 가능하도록 설정"""
        for name, param in self.named_parameters():
            param.requires_grad_(True)
        return

    def init_cvar(self) -> None:
        """CVAR 관련 네트워크 가중치 초기화"""
        for name, param in self.named_parameters():
            if ('encoder') in name or ('decoder' in name):
                torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_cvar(self) -> None:
        """CVAR 관련 네트워크만 학습 가능하도록 설정"""
        for name, param in self.named_parameters():
            if ('encoder' in name) or ('decoder' in name) or ('warmup' in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return

    def warm_item_id(self, x_dict):
        """
        아이템 특징을 활용하여 새로운 아이템 ID 임베딩 생성.

        :param x_dict: 입력 데이터 딕셔너리
        :return: 새로운 아이템 ID 임베딩 (Tensor)
        """
        # 원본 아이템 ID 임베딩
        item_ids = x_dict[self.item_id_name]
        item_id_emb = self.origin_item_emb(item_ids).squeeze()
        
        # 아이템 특징 임베딩 생성
        item_embs = []
        for item_f in self.item_features: 
            type = self.model.description[item_f][1]
            name = f"warmup_{item_f}"
            x = x_dict[item_f]
            
            if type == 'spr':
                emb = self.warmup_emb_layer[name](x).squeeze()
            elif type == 'ctn':
                emb = x
            elif type == 'seq':
                emb = self.warmup_emb_layer[name](x).sum(dim=1, keepdim=True).squeeze()
            elif type == 'emb':
                emb = x
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            
            item_embs.append(emb)
        
        sideinfo_emb = torch.concat(item_embs, dim=1)
        
        # latent space에서 샘플링
        mean = self.mean_encoder(item_id_emb)
        log_v = self.log_v_encoder(item_id_emb)
        mean_p = self.mean_encoder_p(sideinfo_emb)
        log_v_p = self.log_v_encoder_p(sideinfo_emb)
        reg_term = self.wasserstein(mean, log_v, mean_p, log_v_p)
        
        # 노이즈 추가하여 샘플링
        noise = torch.randn(mean.size()).to(self.device)
        z = mean + 1e-4 * torch.exp(log_v * 0.5) * noise
        z_p = mean_p + 1e-4 * torch.exp(log_v_p * 0.5) * noise
        
        # Decoding
        freq = x_dict['count']
        pred = self.decoder(torch.concat([z, freq], 1))
        pred_p = self.decoder(torch.concat([z_p, freq], 1))
        
        # Reconstruction loss
        recon_term = torch.square(pred - item_id_emb).sum(-1).mean()
        
        return pred_p, reg_term, recon_term

    def forward(self, x_dict):
        """
        CVAR 모델의 Forward 과정.

        :param x_dict: 입력 데이터 딕셔너리
        :return: 모델 예측값 (Tensor)
        """
        warm_id_emb, reg_term, recon_term = self.warm_item_id(x_dict)
        target = self.model.forward_with_item_id_emb(warm_id_emb, x_dict)
        return target, recon_term, reg_term

