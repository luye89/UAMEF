import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import datetime
from scipy.stats import pearsonr
import random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from BalancedMSE import ASERLoss
from arguments import args_parser
from utils import is_ascending
from model.model_branch import Enzyme

parser = args_parser()
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

torch.autograd.set_detect_anomaly(False)
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

num_workers = args.num_workers
adjust_step = 3
early_stop = 30

if args.device != "cpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"


def my_collate(batch):
    protein_seqs = [item[0] for item in batch]
    protein_padded = pad_sequence(protein_seqs, batch_first=True, padding_value=0.0)
    protein_mask = (protein_padded.sum(dim=-1) != 0).long()

    mol_seqs = [item[1] for item in batch]
    mol_padded = pad_sequence(mol_seqs, batch_first=True, padding_value=0.0)
    mol_mask = (mol_padded.sum(dim=-1) != 0).long()

    ph = torch.tensor([item[2][0] for item in batch], dtype=torch.float32).unsqueeze(1)
    use_ph = torch.tensor([item[2][1] for item in batch], dtype=torch.long).unsqueeze(1)
    temp = torch.tensor([item[3][0] for item in batch], dtype=torch.float32).unsqueeze(1)
    use_temp = torch.tensor([item[3][1] for item in batch], dtype=torch.long).unsqueeze(1)

    labels = torch.tensor([item[4] for item in batch], dtype=torch.float32).unsqueeze(1)

    return {
        'protein_emb': protein_padded,
        'protein_mask': protein_mask,
        'mol_emb': mol_padded,
        'mol_mask': mol_mask,
        'ph': ph,
        'use_ph': use_ph,
        'temp': temp,
        'use_temp': use_temp,
        'labels': labels
    }


class EnzymeDataset(Dataset):
    def __init__(self, csv_path, prot_h5_path, mol_h5_path, task_name):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.prot_h5_path = prot_h5_path
        self.mol_h5_path = mol_h5_path
        self.prot_h5 = None
        self.mol_h5 = None
        self.task_name = task_name

        self.col_map = {
            'kcat': {'sub_col': 'SMILES', 'label_col': 'log10_kcat'},
            'km': {'sub_col': 'SMILES', 'label_col': 'log10_km'},
            'ki': {'sub_col': 'SMILES', 'label_col': 'log10_ki'}
        }
        config = self.col_map.get(task_name.lower(), self.col_map['kcat'])
        self.target_sub_col = config['sub_col']
        self.target_label_col = config['label_col']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.prot_h5 is None:
            self.prot_h5 = h5py.File(self.prot_h5_path, 'r')
        if self.mol_h5 is None:
            self.mol_h5 = h5py.File(self.mol_h5_path, 'r')

        row = self.df.iloc[idx]
        seq_str = str(row.get('EnzymeSequence'))
        smiles_str = str(row[self.target_sub_col])

        if seq_str in self.prot_h5:
            prot_tensor = torch.from_numpy(self.prot_h5[seq_str][:])
        else:
            prot_tensor = torch.zeros(1, 1024)

        safe_mol_key = smiles_str.replace('/', '_slash_')
        if safe_mol_key in self.mol_h5:
            mol_tensor = torch.from_numpy(self.mol_h5[safe_mol_key][:])
        else:
            mol_tensor = torch.zeros(1, 300)

        raw_ph = row.get('PH')
        if pd.isna(raw_ph) or raw_ph == '':
            ph_val = 0.0
            use_ph = 0
        else:
            ph_val = float(raw_ph)
            use_ph = 1

        raw_temp = row.get('Temperature')
        if pd.isna(raw_temp) or raw_temp == '':
            temp_val = 0.0
            use_temp = 0
        else:
            temp_val = float(raw_temp)
            use_temp = 1

        label_val = float(row[self.target_label_col])
        label = torch.tensor(label_val, dtype=torch.float32)

        return prot_tensor, mol_tensor, (ph_val, use_ph), (temp_val, use_temp), label


def get_enzyme_datasets(args, task, prot_path, mol_path):
    task_dir = os.path.join(args.data_dir, task)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
        print(f"Created directory: {task_dir}")

    train_file = os.path.join(task_dir, 'train.csv')
    val_file = os.path.join(task_dir, 'val.csv')
    test_file = os.path.join(task_dir, 'test.csv')

    if not (os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file)):
        print(f"Task [{task}]: Split files not found. Generating new splits...")
        origin_csv = os.path.join(args.data_dir, f"{task}.csv")
        df = pd.read_csv(origin_csv)
        train_df, test_df = train_test_split(df, test_size=0.6, random_state=args.seed)
        val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=args.seed)
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
    else:
        print(f"Task [{task}]: Found existing split files. Loading directly.")

    train_ds = EnzymeDataset(train_file, prot_path, mol_path, task)
    val_ds = EnzymeDataset(val_file, prot_path, mol_path, task)
    test_ds = EnzymeDataset(test_file, prot_path, mol_path, task)
    return train_ds, val_ds, test_ds


def main():
    prot_feat_path = r"D:\data\prot_embedding.h5"
    mol_feat_path = r"D:\data\mol_embedding_un.h5"

    # 解析多个任务，支持 'kcat,km'
    raw_task = args.task.split(',') if isinstance(args.task, str) else [args.task]

    if not os.path.exists('checkpoints'): os.mkdir('checkpoints')
    if not os.path.exists('log'): os.mkdir('log')

    # 外层循环：遍历每一个任务
    for task_now in raw_task:
        task_now = task_now.strip()
        if not task_now:
            continue

        print(f"\n{'=' * 60}")
        print(f" 开始执行任务：{task_now.upper()} ")
        print(f"{'=' * 60}")

        no_improve_counter = 0

        # 获取单一任务数据集
        train_ds, val_ds, test_ds = get_enzyme_datasets(args, task_now, prot_feat_path, mol_feat_path)
        train_targets = train_ds.df[train_ds.target_label_col].values

        criterion_train = ASERLoss(all_targets_of_train_set=train_targets, k=1)

        train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=my_collate,
                              pin_memory=True, num_workers=args.num_workers, persistent_workers=True, drop_last=True)
        val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=my_collate,
                            pin_memory=True, num_workers=args.num_workers, persistent_workers=True, drop_last=False)
        test_dl = DataLoader(test_ds, args.batch_size, shuffle=False, collate_fn=my_collate,
                             pin_memory=True, num_workers=args.num_workers, persistent_workers=True, drop_last=False)

        # 每轮任务重新初始化模型，保证参数完全隔离
        model = Enzyme(hidden_dim=args.hidden_dim, expert_total=10, device=device).to(device)
        criterion_val = torch.nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, capturable=True)

        current_lr = args.lr
        val_record = []
        best_epoch = 0
        best_valid_res = float('inf')

        # 加上 _z 的日志文件名
        file_path = f'log/[{args.model}]_{task_now}_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.txt'

        # 断点文件名加入 task_now 和 _z 防止覆盖
        checkpoint_path = f'checkpoints/latest_resume_checkpoint_{task_now}.pt'
        start_epoch = 1

        if os.path.exists(checkpoint_path):
            print(f" 发现中断的训练记录 {checkpoint_path}，正在恢复状态...")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            current_lr = checkpoint.get('current_lr', current_lr)
            best_epoch = checkpoint['best_epoch']
            best_valid_res = checkpoint['best_valid_res']
            val_record = checkpoint['val_record']
            no_improve_counter = checkpoint.get('no_improve_counter', 0)
            if 'file_path' in checkpoint: file_path = checkpoint['file_path']

            print(f"成功恢复！将从第 {start_epoch} 轮继续训练...")
            print(f"   当前最佳 Epoch: {best_epoch}, 最佳 RMSE: {best_valid_res:.4f}")

        for epoch in tqdm(range(start_epoch, args.epochs + 1)):
            print('Epoch:', epoch)
            model.train()
            train_loss_sum = 0
            total_steps = 0
            optimizer.param_groups[0]['lr'] = current_lr

            print(f'Task: {task_now} Training!')

            with tqdm(train_dl, position=0, ncols=150, colour='#666666') as tqdm_range:
                for i, batch_data in enumerate(tqdm_range):
                    optimizer.zero_grad()

                    prot_emb = batch_data['protein_emb'].to(device)
                    prot_mask = batch_data['protein_mask'].to(device)
                    mol_emb = batch_data['mol_emb'].to(device)
                    mol_mask = batch_data['mol_mask'].to(device)
                    ph = batch_data['ph'].to(device)
                    use_ph = batch_data['use_ph'].to(device)
                    temp = batch_data['temp'].to(device)
                    use_temp = batch_data['use_temp'].to(device)
                    labels = batch_data['labels'].to(device)

                    y_pred, ortho_loss, moe_loss = model(prot_emb, prot_mask, mol_emb, mol_mask, ph, use_ph, temp,
                                                         use_temp)

                    raw_loss = criterion_train(y_pred, labels)

                    with torch.no_grad():
                        pure_mse = F.mse_loss(y_pred, labels)

                    final_loss = raw_loss + ortho_loss * 0.5 + moe_loss
                    train_loss_sum += pure_mse.item()
                    total_steps += 1

                    final_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            avg_train_loss = train_loss_sum / total_steps if total_steps > 0 else 0.0
            print(f'Train loss (MSE): {avg_train_loss:.4f}')

            with open(file_path, "a", encoding='utf-8') as f:
                f.write(f'Epoch: {epoch}\n')
                f.write(f'lr: {current_lr}\n')
                f.write(f'Train loss (MSE): {avg_train_loss:.4f}\n')

            # Valid
            with torch.no_grad():
                model.eval()
                val_loss = 0
                all_preds = []
                all_labels = []

                with tqdm(val_dl, position=0, ncols=150, colour='#666666', desc=f'Valid {task_now}') as tqdm_range:
                    for i, batch_data in enumerate(tqdm_range):
                        prot_emb = batch_data['protein_emb'].to(device)
                        prot_mask = batch_data['protein_mask'].to(device)
                        mol_emb = batch_data['mol_emb'].to(device)
                        mol_mask = batch_data['mol_mask'].to(device)
                        ph = batch_data['ph'].to(device)
                        use_ph = batch_data['use_ph'].to(device)
                        temp = batch_data['temp'].to(device)
                        use_temp = batch_data['use_temp'].to(device)
                        labels = batch_data['labels'].to(device)

                        y_pred, _, _ = model(prot_emb, prot_mask, mol_emb, mol_mask, ph, use_ph, temp, use_temp)

                        loss = criterion_val(y_pred, labels)
                        val_loss += loss.item()

                        all_preds.append(y_pred.detach().cpu().numpy())
                        all_labels.append(labels.detach().cpu().numpy())

                avg_val_loss = val_loss / len(val_dl)
                val_record.append(avg_val_loss)

                all_preds = np.concatenate(all_preds, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)

                val_mse = mean_squared_error(all_labels, all_preds)
                val_rmse = np.sqrt(val_mse)
                val_r2 = r2_score(all_labels, all_preds)
                val_mae = mean_absolute_error(all_labels, all_preds)
                val_pcc = pearsonr(all_labels.flatten(), all_preds.flatten())[0] if len(all_preds) > 1 else 0.0

                print(
                    f'Task: {task_now} Valid RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | R2: {val_r2:.4f} | PCC: {val_pcc:.4f} | Loss: {avg_val_loss:.4f}')

                with open(file_path, "a", encoding='utf-8') as f:
                    f.write(
                        f'Task: {task_now} Valid RMSE:{val_rmse:.4f} MAE:{val_mae:.4f} R2:{val_r2:.4f} PCC:{val_pcc:.4f} Loss:{avg_val_loss:.4f}\n')

                if val_rmse < best_valid_res:
                    best_epoch = epoch
                    best_valid_res = val_rmse
                    # 保存最佳模型，文件名会自动带有 _z
                    torch.save(model.state_dict(), 'checkpoints/' + file_path.split('/')[-1][:-4] + '.pt')
                    print(f"New Best Model Saved! (RMSE: {best_valid_res:.4f})")
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1

                if len(val_record) >= adjust_step:
                    if is_ascending(val_record[-adjust_step:]):
                        current_lr = max(current_lr * 0.5, 1e-6)

            # 断点保存
            resume_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'current_lr': current_lr,
                'best_epoch': best_epoch,
                'best_valid_res': best_valid_res,
                'val_record': val_record,
                'no_improve_counter': no_improve_counter,
                'file_path': file_path
            }
            torch.save(resume_checkpoint, checkpoint_path)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if no_improve_counter > early_stop:
                print(f"Early stopping triggered after {early_stop} epochs with no improvement.")
                break

        print(f'Best Epoch: {best_epoch} Best Val RMSE: {best_valid_res:.4f}')
        with open(file_path, "a", encoding='utf-8') as f:
            f.write(f'Best Epoch:{best_epoch} Best Val RMSE:{best_valid_res:.4f}\n')

        print(f"\nStarting Testing with Best Model for Task {task_now}...")
        with torch.no_grad():
            # 获取加载模型的路径，会自动包含 _z
            model_path = 'checkpoints/' + file_path.split('/')[-1][:-4] + '.pt'
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                print(f"Loaded model from {model_path}")
            else:
                print("Warning: Best model not found, using current model.")

            model.eval()
            all_preds = []
            all_labels = []

            with tqdm(test_dl, position=0, ncols=150, colour='#666666', desc=f'Test {task_now}') as tqdm_range:
                for i, batch_data in enumerate(tqdm_range):
                    prot_emb = batch_data['protein_emb'].to(device)
                    prot_mask = batch_data['protein_mask'].to(device)
                    mol_emb = batch_data['mol_emb'].to(device)
                    mol_mask = batch_data['mol_mask'].to(device)
                    ph = batch_data['ph'].to(device)
                    use_ph = batch_data['use_ph'].to(device)
                    temp = batch_data['temp'].to(device)
                    use_temp = batch_data['use_temp'].to(device)
                    labels = batch_data['labels'].to(device)

                    y_pred, _, _ = model(prot_emb, prot_mask, mol_emb, mol_mask, ph, use_ph, temp, use_temp)

                    all_preds.append(y_pred.detach().cpu().numpy())
                    all_labels.append(labels.detach().cpu().numpy())

            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            test_mse = mean_squared_error(all_labels, all_preds)
            test_rmse = np.sqrt(test_mse)
            test_r2 = r2_score(all_labels, all_preds)
            test_mae = mean_absolute_error(all_labels, all_preds)
            test_pcc = pearsonr(all_labels.flatten(), all_preds.flatten())[0] if len(all_preds) > 1 else 0.0

            print(
                f'Test Result - Task: {task_now} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R2: {test_r2:.4f} | PCC: {test_pcc:.4f}')

            with open(file_path, "a", encoding='utf-8') as f:
                f.write(
                    f'Task: {task_now} Test RMSE:{test_rmse:.4f} MAE:{test_mae:.4f} R2:{test_r2:.4f} PCC:{test_pcc:.4f}\n')

        # 当前任务结束，强制清空模型和显存，确保下一个任务环境纯净
        del model
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()