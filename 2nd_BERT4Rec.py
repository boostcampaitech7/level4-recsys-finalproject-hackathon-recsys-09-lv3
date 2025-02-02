import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForMaskedLM
from sklearn.metrics import ndcg_score
from collections import defaultdict
from datetime import datetime
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def preprocess_data(save_dir="preprocessed_data", sample_size=10000, random_state=42):
    os.makedirs(save_dir, exist_ok=True)
    
    # 저장 파일 경로 수정
    preprocessed_files = {
        'user_sequences': os.path.join(save_dir, "user_sequences.json"),
        'app_to_idx': os.path.join(save_dir, "app_to_idx.json")
    }

    # 저장된 데이터 존재 시 로드
    if all(os.path.exists(file) for file in preprocessed_files.values()):
        print("Loading preprocessed data from disk...")
        with open(preprocessed_files['user_sequences'], "r") as f:
            user_sequences = {int(k): v for k, v in json.load(f).items()}
        with open(preprocessed_files['app_to_idx'], "r") as f:
            app_to_idx = {int(k): v for k, v in json.load(f).items()}
    else:
        # 원본 데이터 처리
        print("Preprocessing data...")
        df = pd.read_csv('recommendations.csv')
        
        # app_id 매핑
        unique_apps = df['app_id'].unique()
        app_to_idx = {int(app): i+1 for i, app in enumerate(unique_apps)}
        df['app_id'] = df['app_id'].map(lambda x: app_to_idx[int(x)])

        # 유저 시퀀스 생성
        user_sequences = defaultdict(list)
        df = df.sort_values(['user_id', 'date'])
        for user, group in df.groupby('user_id'):
            user_sequences[user] = group['app_id'].tolist()

        # 데이터 저장
        print("Saving raw preprocessed data...")
        with open(preprocessed_files['user_sequences'], "w") as f:
            json.dump({str(k): v for k, v in user_sequences.items()}, f)
        with open(preprocessed_files['app_to_idx'], "w") as f:
            json.dump({str(k): v for k, v in app_to_idx.items()}, f)

    # 필터링 및 샘플링 (저장된 데이터에도 동일 적용)
    print("Applying sequence filtering...")
    sequence_lengths = [len(seq) for seq in user_sequences.values()]
    #length_threshold = np.percentile(sequence_lengths, 99)
    filtered_sequences = {u: s for u, s in user_sequences.items() 
                         if len(s) >= 5}

    max_seq_len = max(len(seq) for seq in filtered_sequences.values())
    num_items = len(app_to_idx)  # app_to_idx 크기로 계산

    '''# 샘플링
    if len(filtered_sequences) > sample_size:
        rng = np.random.RandomState(random_state)
        sampled_users = rng.choice(list(filtered_sequences.keys()), 
                                 size=sample_size, replace=False)
        filtered_sequences = {u: filtered_sequences[u] for u in sampled_users}'''

    # 최종 출력 형식
    user_ids = list(filtered_sequences.keys())
    user_actions = list(filtered_sequences.values())
    
    return user_ids, user_actions, max_seq_len, num_items, app_to_idx

# 2. 데이터 로더 구현 (수정) ===================================================
class BERT4RecDataset(Dataset):
    def __init__(self, user_ids, user_actions, num_items, max_len, mask_prob=0.15, random_state=42, mode='train'):
        self.user_ids = user_ids  # ✅ 사용자 ID 저장
        self.user_actions = user_actions
        self.num_items = num_items
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = num_items + 1
        self.interaction_counts = [len(seq) for seq in user_actions]  # 상호작용 횟수 저장
        self.rng = np.random.RandomState(random_state)  # RandomState 추가
        self.mode = mode

    def __len__(self):
        return len(self.user_actions)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]  # ✅ 사용자 ID 추출
        sequence = self.user_actions[idx]
        
        # 슬라이딩 윈도우 (Train만 적용)
        if self.mode == 'train':
            # 모든 가능한 서브 시퀀스 생성
            start_idx = max(0, len(sequence) - self.max_len)
            windows = [sequence[i:i+self.max_len] for i in range(start_idx)]
            if not windows:  # 시퀀스가 max_len보다 짧은 경우
                windows = [sequence]
            window = windows[self.rng.choice(len(windows))]  # 랜덤 샘플링
        else:
            # Valid/Test: 마지막 max_len 아이템 사용
            window = sequence[-self.max_len:]
        
        # 패딩 적용
        padded_seq = self._pad_sequence(window)
        masked_seq, labels = self._apply_masking(padded_seq)
        
        return {
            'user_id': user_id,  # ✅ 사용자 ID 반환
            'input_ids': torch.LongTensor(masked_seq),
            'labels': torch.LongTensor(labels),
            'attention_mask': torch.LongTensor([1 if x !=0 else 0 for x in masked_seq]),
            'interaction_count': self.interaction_counts[idx]
        }
    def _pad_sequence(self, sequence):
        """시퀀스를 max_len 길이로 패딩합니다."""
        if len(sequence) < self.max_len:
            # 짧은 시퀀스: 앞쪽을 패딩 토큰으로 채움
            padded_seq = [0] * (self.max_len - len(sequence)) + sequence
        else:
            # 긴 시퀀스: max_len만큼 자름
            padded_seq = sequence[-self.max_len:]
        return padded_seq
        
    def _apply_masking(self, seq):
        masked_seq = seq.copy()
        labels = [-100] * len(seq)
        
        if self.mode == 'train':
            # 랜덤 마스킹 (마지막 위치 제외)
            for i in range(len(seq)-1):
                if self.rng.rand() < self.mask_prob and seq[i] != 0:
                    labels[i] = seq[i]
                    masked_seq[i] = self.mask_token
        else:
            # Valid/Test: 마지막 위치만 마스킹
            labels[-1] = seq[-1]
            masked_seq[-1] = self.mask_token
        
        return masked_seq, labels

# 3. 모델 구현 (수정) ================================================================
class BERT4Rec(torch.nn.Module):
    def __init__(self, num_items, max_seq_len, hyperparams, pad_token=0, mask_token=None):
        super().__init__()
        self.pad_token = pad_token
        self.mask_token = mask_token if mask_token else num_items + 1
        
        self.bert_config = BertConfig(
            vocab_size=num_items + 2,  # [아이템] + [PAD] + [MASK]
            hidden_size=hyperparams['hidden_size'],
            num_hidden_layers=hyperparams['num_layers'],
            num_attention_heads=hyperparams['num_heads'],
            intermediate_size=hyperparams['hidden_size']*4,
            max_position_embeddings=max_seq_len,
            attention_probs_dropout_prob=hyperparams['attention_probs_dropout'],
            hidden_dropout_prob=hyperparams['hidden_dropout']
        )
        self.bert = BertForMaskedLM(self.bert_config).to(device)
    
    def forward(self, input_ids, attention_mask, labels):
        return self.bert(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            labels=labels.to(device)
        )
    
def save_evaluation_results(data, save_path):
    os.makedirs(save_path, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_path, "evaluation_results.csv"), index=False)
    print(f"✅ Results saved to {save_path}")

def calculate_overall_metrics(results):
    ndcg_scores = []
    recall_scores = []
    for _, true, pred in results:
        hit = true in pred
        recall_scores.append(hit)
        ndcg_scores.append(1 / np.log2(pred.index(true)+2) if hit else 0)
    
    return {
        'ndcg': np.mean(ndcg_scores),
        'recall': np.mean(recall_scores)
    }

def calculate_group_metrics(results, bins, group_names):
    group_metrics = {}
    for name, lower, upper in zip(group_names, bins[:-1], bins[1:]):
        group_data = [(ic, t, p) for ic, t, p in results if lower <= ic < upper]
        if group_data:
            group_metrics[name] = calculate_overall_metrics(group_data)
    return group_metrics

# 4. 평가 메트릭 (수정) =======================================================
def evaluate(model, dataloader, idx_to_app, k=10, group_eval=False, save_dir=None):
    model.eval()
    results = []
    total_loss = 0.0
    num_batches = 0
    eval_data = defaultdict(list)
    
    def get_interaction_group(count):
        try:
            return next(
                name for name, l, u in zip(GROUP_NAMES, BINS[:-1], BINS[1:])
                if l <= count < u
            )
        except StopIteration:
            return 'unknown'
        
    # 1. 공통 상수 정의
    BINS = [5, 10, 20, 30, 40, 50]
    GROUP_NAMES = ['5-10', '10-20', '20-30', '30-40', '40-50']

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
            
            # [수정] 마지막 위치 로그잇만 선택
            last_pos_logits = outputs.logits[:, -1, :]  # (batch_size, num_items)
            
            # [수정] 아이템 제외 로직 개선
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                current_input = input_ids[i].tolist()
                collected_items = list(set([x for x in current_input 
                                          if x not in [0, model.mask_token]]))  # 0과 MASK 제외
                if collected_items:
                    last_pos_logits[i, collected_items] = -1e10
            
            preds = last_pos_logits.argsort(dim=-1, descending=True)
            targets = labels[:, -1].cpu().numpy()  # 마지막 위치 타깃
            
            # [수정] 결과 처리
            for i in range(batch_size):
                user_id = batch['user_id'][i].item()  # ✅ 사용자 ID 추출
                true_idx = targets[i]
                pred_indices = preds[i][:k].tolist()
                interaction_count = batch['interaction_count'][i].item() if batch['interaction_count'][i].item() < 50 else 50
                
                true_app = idx_to_app.get(true_idx, 0)
                pred_apps = [idx_to_app.get(p, 0) for p in pred_indices]

                # 그룹 정보 계산
                group = get_interaction_group(interaction_count)

                eval_data['user_id'].append(user_id)
                eval_data['true_app'].append(true_app)
                eval_data['predicted_apps'].append(pred_apps)
                eval_data['interaction_count'].append(interaction_count)
                eval_data['group'].append(group)

                results.append((interaction_count, true_app, pred_apps))
    if save_dir:
        save_evaluation_results(eval_data, save_dir)
    loss = total_loss / num_batches
    overall_metrics = calculate_overall_metrics(results)
    overall_metrics['loss'] = loss
    group_metrics = calculate_group_metrics(results, BINS, GROUP_NAMES) if group_eval else {}

    return {
        'overall': overall_metrics,
        'group_metrics': group_metrics,
        'eval_data': pd.DataFrame(eval_data) if save_dir else None
    }




class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def update(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            return True  # 개선됨
        else:
            self.counter += 1
            return False

    def should_stop(self):
        return self.counter >= self.patience

# 5. 실행 파이프라인 ==========================================================
def main(random_state=42):
    # Random seed 설정
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
        
    # 데이터 로드
    print("data loaded")
    # 전처리
    user_ids, user_actions, max_seq_len, num_items, app_to_idx = preprocess_data(sample_size=10000, random_state=random_state)
    print(f"Total sampled users: {len(user_ids)}")  # 샘플링 사용자 수 확인
    print(f"Max sequence length: {max_seq_len}")    # 시퀀스 길이 확인
    print(f"Total items: {num_items}")              # 아이템 수 확인
    print("data preprocessed")
    max_seq_len = 50
    
    # idx_to_app 생성
    idx_to_app = {v: k for k, v in app_to_idx.items()}

    # 데이터셋 생성
    train_user_actions, val_user_actions, test_user_actions = [], [], []
    for seq in user_actions:
        if len(seq) >= 3:  # 최소 3개 아이템 필요
            train_user_actions.append(seq[:-2])
            val_user_actions.append(seq[:-1])  # valid 타깃: seq[-2]
            test_user_actions.append(seq)      # test 타깃: seq[-1]
    print("data split")

    mask_token = num_items + 1
    pad_token = 0

    # Create datasets and dataloaders
    train_dataset = BERT4RecDataset(
        user_ids=user_ids,  # ✅ 사용자 ID 추가
        user_actions=train_user_actions,
        num_items=num_items,
        max_len=max_seq_len,
        mask_prob=0.15,
        mode='train',
        random_state=random_state
    )
    
    val_dataset = BERT4RecDataset(
        user_ids=user_ids,  # ✅ 사용자 ID 추가
        user_actions=val_user_actions,
        num_items=num_items,
        max_len=max_seq_len,
        mask_prob=0.0,
        mode='valid',
        random_state=random_state
    )
    
    test_dataset = BERT4RecDataset(
        user_ids=user_ids,
        user_actions=test_user_actions,
        num_items=num_items,
        max_len=max_seq_len,
        mask_prob=0.0,
        mode='test',
        random_state=random_state
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True)
    
    log_dir = f"runs/exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    # 하이퍼파라미터 저장
    hyperparams = {
        'hidden_size': 128,
        'num_layers': 3,
        'num_heads': 4,
        'attention_probs_dropout': 0.2,
        'hidden_dropout': 0.2, 
        'mask_prob': 0.2 
    }
        
    # 모델 초기화 및 GPU 이동
    model = BERT4Rec(
        num_items, 
        max_seq_len, 
        hyperparams, 
        pad_token=pad_token,
        mask_token=mask_token
    )
    print("model initialized")
    # 학습 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # Recall 최대화 방향
        patience=2,
        factor=0.5,
        verbose=True
    )    
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    
    # 결과 저장을 위한 디렉토리 생성
    save_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    
    with open(f"{save_dir}/hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=4)

    # 학습 루프 개선
    best_model_path = f"{save_dir}/best_checkpoint.pt"
    early_stopping = EarlyStopper(patience=3, min_delta=0.001)

    # 학습 루프
    for epoch in range(30):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch+1}/30")
        print(f"{'='*40}")
        
        # 학습 단계
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        # 진행률 표시줄 추가
        with tqdm(train_dataloader, unit="batch", desc=f"Training") as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                
                # GPU로 데이터 이동
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                # 혼합 정밀도 학습
                with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                    outputs = model(input_ids, attention_mask, labels)
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # 메트릭 업데이트
                batch_loss = loss.item()
                epoch_loss += batch_loss
                batch_count += 1
                
                # 진행률 표시줄 업데이트
                pbar.set_postfix({
                    'loss': f"{batch_loss:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # 텐서보드에 배치별 loss 기록
                writer.add_scalar('Loss/train_batch', batch_loss, epoch*len(train_dataloader)+batch_count)
        
        # 에폭별 평균 loss 계산
        avg_epoch_loss = epoch_loss / batch_count
        print(f"\nTrain Loss: {avg_epoch_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
        
        # 검증 단계 (전체 평가만 수행)
        eval_result = evaluate(
            model, 
            val_dataloader, 
            idx_to_app,
            save_dir=save_dir  # ✅ 결과 저장 경로 추가
        )
        val_metrics = eval_result['overall']
        
        print(f"\nValidation Metrics:")
        print(f"- Loss: {val_metrics['loss']:.4f}")
        print(f"- NDCG@10: {val_metrics['ndcg']:.4f}")
        print(f"- Recall@10: {val_metrics['recall']:.4f}")
        
        # Early Stopping 및 모델 저장
        if early_stopping.update(val_metrics['loss']):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, best_model_path)
            print(f"Best model saved at epoch {epoch+1}")
            
        if early_stopping.should_stop():
            print("Early stopping triggered!")
            break

    # 검증, 테스트, 최종 평가 수행
    val_result = evaluate(
        model, val_dataloader, idx_to_app, 
        save_dir=os.path.join(save_dir, "validation")
    )
    test_result = evaluate(
        model, test_dataloader, idx_to_app, 
        save_dir=os.path.join(save_dir, "test")
    )
    
    # Best Model 로드 (보안 경고 해결)
    best_model_path = os.path.join(save_dir, "best_checkpoint.pt")
    best_checkpoint = torch.load(best_model_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # 최종 테스트 평가
    final_test_result = evaluate(
        model, test_dataloader, idx_to_app, 
        save_dir=os.path.join(save_dir, "final_test")
    )
    
    # 최종 평가 (그룹별 메트릭 포함)
    print("\nFinal Evaluation with Best Model:")
    best_checkpoint = torch.load(best_model_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # 평가 시 idx_to_app 전달
    final_result = evaluate(model, test_dataloader, idx_to_app, group_eval=True)
    
    # 결과 저장 시 원본 app_id 사용
    with open(f"{save_dir}/final_metrics.json", "w") as f:
        json.dump(final_result, f, indent=4)

    print("\nOverall Metrics:")
    print(f"- NDCG@10: {final_result['overall']['ndcg']:.4f}")
    print(f"- Recall@10: {final_result['overall']['recall']:.4f}")
    
    print("\nGroup-wise Metrics:")
    for group, metrics in final_result['group_metrics'].items():
        print(f"Group {group}:")
        print(f"  NDCG@10: {metrics['ndcg']:.4f}")
        print(f"  Recall@10: {metrics['recall']:.4f}")

if __name__ == "__main__":
    main()