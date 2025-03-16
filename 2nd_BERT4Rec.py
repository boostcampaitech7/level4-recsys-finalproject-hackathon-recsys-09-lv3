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
    
    preprocessed_files = {
        'user_sequences': os.path.join(save_dir, "user_sequences.json"),
        'app_to_idx': os.path.join(save_dir, "app_to_idx.json")
    }

    if all(os.path.exists(file) for file in preprocessed_files.values()):
        print("Loading preprocessed data from disk...")
        with open(preprocessed_files['user_sequences'], "r") as f:
            user_sequences = {int(k): v for k, v in json.load(f).items()}
        with open(preprocessed_files['app_to_idx'], "r") as f:
            app_to_idx = {int(k): v for k, v in json.load(f).items()}
    else:
        print("Preprocessing data...")
        df = pd.read_csv('recommendations.csv')
        
        unique_apps = df['app_id'].unique()
        app_to_idx = {int(app): i+1 for i, app in enumerate(unique_apps)}
        df['app_id'] = df['app_id'].map(lambda x: app_to_idx[int(x)])

        user_sequences = defaultdict(list)
        df = df.sort_values(['user_id', 'date'])
        for user, group in df.groupby('user_id'):
            user_sequences[user] = group['app_id'].tolist()

        print("Saving raw preprocessed data...")
        with open(preprocessed_files['user_sequences'], "w") as f:
            json.dump({str(k): v for k, v in user_sequences.items()}, f)
        with open(preprocessed_files['app_to_idx'], "w") as f:
            json.dump({str(k): v for k, v in app_to_idx.items()}, f)

    print("Applying sequence filtering...")
    sequence_lengths = [len(seq) for seq in user_sequences.values()]
    filtered_sequences = {u: s for u, s in user_sequences.items() 
                         if len(s) >= 5}

    max_seq_len = max(len(seq) for seq in filtered_sequences.values())
    num_items = len(app_to_idx)

    user_ids = list(filtered_sequences.keys())
    user_actions = list(filtered_sequences.values())
    
    return user_ids, user_actions, max_seq_len, num_items, app_to_idx

class BERT4RecDataset(Dataset):
    def __init__(self, user_ids, user_actions, num_items, max_len, mask_prob=0.15, random_state=42, mode='train'):
        self.user_ids = user_ids
        self.user_actions = user_actions
        self.num_items = num_items
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = num_items + 1
        self.interaction_counts = [len(seq) for seq in user_actions]
        self.rng = np.random.RandomState(random_state)
        self.mode = mode

    def __len__(self):
        return len(self.user_actions)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        sequence = self.user_actions[idx]
        
       if self.mode == 'train':
           start_idx = max(0, len(sequence) - self.max_len)
            windows = [sequence[i:i+self.max_len] for i in range(start_idx)]
            if not windows:
                windows = [sequence]
            window = windows[self.rng.choice(len(windows))]
        else:
            window = sequence[-self.max_len:]
        
        padded_seq = self._pad_sequence(window)
        masked_seq, labels = self._apply_masking(padded_seq)
        
        return {
            'user_id': user_id,
            'input_ids': torch.LongTensor(masked_seq),
            'labels': torch.LongTensor(labels),
            'attention_mask': torch.LongTensor([1 if x !=0 else 0 for x in masked_seq]),
            'interaction_count': self.interaction_counts[idx]
        }
    def _pad_sequence(self, sequence):
        """시퀀스를 max_len 길이로 패딩합니다."""
        if len(sequence) < self.max_len:
            padded_seq = [0] * (self.max_len - len(sequence)) + sequence
        else:
           padded_seq = sequence[-self.max_len:]
        return padded_seq
        
    def _apply_masking(self, seq):
        masked_seq = seq.copy()
        labels = [-100] * len(seq)
        
        if self.mode == 'train':
            for i in range(len(seq)-1):
                if self.rng.rand() < self.mask_prob and seq[i] != 0:
                    labels[i] = seq[i]
                    masked_seq[i] = self.mask_token
        else:
            labels[-1] = seq[-1]
            masked_seq[-1] = self.mask_token
        
        return masked_seq, labels

class BERT4Rec(torch.nn.Module):
    def __init__(self, num_items, max_seq_len, hyperparams, pad_token=0, mask_token=None):
        super().__init__()
        self.pad_token = pad_token
        self.mask_token = mask_token if mask_token else num_items + 1
        
        self.bert_config = BertConfig(
            vocab_size=num_items + 2,
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
    print(f"Results saved to {save_path}")

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
            
           last_pos_logits = outputs.logits[:, -1, :]
            
           batch_size = input_ids.size(0)
            for i in range(batch_size):
                current_input = input_ids[i].tolist()
                collected_items = list(set([x for x in current_input 
                                          if x not in [0, model.mask_token]]))
                if collected_items:
                    last_pos_logits[i, collected_items] = -1e10
            
            preds = last_pos_logits.argsort(dim=-1, descending=True)
            targets = labels[:, -1].cpu().numpy()
            
           for i in range(batch_size):
                user_id = batch['user_id'][i].item()
                true_idx = targets[i]
                pred_indices = preds[i][:k].tolist()
                interaction_count = batch['interaction_count'][i].item() if batch['interaction_count'][i].item() < 50 else 50
                
                true_app = idx_to_app.get(true_idx, 0)
                pred_apps = [idx_to_app.get(p, 0) for p in pred_indices]

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
            return True
        else:
            self.counter += 1
            return False

    def should_stop(self):
        return self.counter >= self.patience

def main(random_state=42):
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
        
   print("data loaded")
    user_ids, user_actions, max_seq_len, num_items, app_to_idx = preprocess_data(sample_size=10000, random_state=random_state)
    print(f"Total sampled users: {len(user_ids)}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Total items: {num_items}")
    print("data preprocessed")
    max_seq_len = 50
    
    idx_to_app = {v: k for k, v in app_to_idx.items()}

   train_user_actions, val_user_actions, test_user_actions = [], [], []
    for seq in user_actions:
        if len(seq) >= 3:
            train_user_actions.append(seq[:-2])
            val_user_actions.append(seq[:-1])
            test_user_actions.append(seq)
    print("data split")

    mask_token = num_items + 1
    pad_token = 0

    train_dataset = BERT4RecDataset(
        user_ids=user_ids,
        user_actions=train_user_actions,
        num_items=num_items,
        max_len=max_seq_len,
        mask_prob=0.15,
        mode='train',
        random_state=random_state
    )
    
    val_dataset = BERT4RecDataset(
        user_ids=user_ids,
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

   hyperparams = {
        'hidden_size': 128,
        'num_layers': 3,
        'num_heads': 4,
        'attention_probs_dropout': 0.2,
        'hidden_dropout': 0.2, 
        'mask_prob': 0.2 
    }
        
    model = BERT4Rec(
        num_items, 
        max_seq_len, 
        hyperparams, 
        pad_token=pad_token,
        mask_token=mask_token
    )
    print("model initialized")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        patience=2,
        factor=0.5,
        verbose=True
    )    
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    
    save_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    
    with open(f"{save_dir}/hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=4)

    best_model_path = f"{save_dir}/best_checkpoint.pt"
    early_stopping = EarlyStopper(patience=3, min_delta=0.001)

    for epoch in range(30):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch+1}/30")
        print(f"{'='*40}")
        
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        with tqdm(train_dataloader, unit="batch", desc=f"Training") as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                    outputs = model(input_ids, attention_mask, labels)
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                batch_loss = loss.item()
                epoch_loss += batch_loss
                batch_count += 1
                
               pbar.set_postfix({
                    'loss': f"{batch_loss:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
        
        avg_epoch_loss = epoch_loss / batch_count
        print(f"\nTrain Loss: {avg_epoch_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
        
        eval_result = evaluate(
            model, 
            val_dataloader, 
            idx_to_app,
            save_dir=save_dir
        )
        val_metrics = eval_result['overall']
        
        print(f"\nValidation Metrics:")
        print(f"- Loss: {val_metrics['loss']:.4f}")
        print(f"- NDCG@10: {val_metrics['ndcg']:.4f}")
        print(f"- Recall@10: {val_metrics['recall']:.4f}")
        
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

    val_result = evaluate(
        model, val_dataloader, idx_to_app, 
        save_dir=os.path.join(save_dir, "validation")
    )
    test_result = evaluate(
        model, test_dataloader, idx_to_app, 
        save_dir=os.path.join(save_dir, "test")
    )
    
    best_model_path = os.path.join(save_dir, "best_checkpoint.pt")
    best_checkpoint = torch.load(best_model_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    final_test_result = evaluate(
        model, test_dataloader, idx_to_app, 
        save_dir=os.path.join(save_dir, "final_test")
    )
    
    print("\nFinal Evaluation with Best Model:")
    best_checkpoint = torch.load(best_model_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    final_result = evaluate(model, test_dataloader, idx_to_app, group_eval=True)
    
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
