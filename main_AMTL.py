import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import datetime
from scipy.stats import pearsonr
import random
import numpy as np
import pandas as pd
import h5py
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from BalancedMSE import ASERLoss
from arguments import args_parser
from utils import is_ascending
from model.model_AMTL import Enzyme

parser = args_parser()
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

torch.autograd.set_detect_anomaly(False)
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
early_stop = 30


def unified_collate(batch):
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

    labels = torch.stack([item[4] for item in batch], dim=0)  # [Batch, 3]
    masks = torch.stack([item[5] for item in batch], dim=0)  # [Batch, 3]

    return {
        'protein_emb': protein_padded, 'protein_mask': protein_mask,
        'mol_emb': mol_padded, 'mol_mask': mol_mask,
        'ph': ph, 'use_ph': use_ph,
        'temp': temp, 'use_temp': use_temp,
        'labels': labels, 'masks': masks
    }


class UnifiedEnzymeDataset(Dataset):
    def __init__(self, csv_path, prot_h5_path, mol_h5_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.prot_h5_path = prot_h5_path
        self.mol_h5_path = mol_h5_path
        self.prot_h5, self.mol_h5 = None, None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.prot_h5 is None: self.prot_h5 = h5py.File(self.prot_h5_path, 'r')
        if self.mol_h5 is None: self.mol_h5 = h5py.File(self.mol_h5_path, 'r')

        row = self.df.iloc[idx]
        seq_str = str(row.get('EnzymeSequence'))
        smiles_str = str(row.get('SMILES'))

        prot_tensor = torch.from_numpy(self.prot_h5[seq_str][:]) if seq_str in self.prot_h5 else torch.zeros(1, 1024)
        safe_mol_key = smiles_str.replace('/', '_slash_')
        mol_tensor = torch.from_numpy(self.mol_h5[safe_mol_key][:]) if safe_mol_key in self.mol_h5 else torch.zeros(1,
                                                                                                                    300)

        raw_ph = row.get('PH')
        ph_val, use_ph = (0.0, 0) if pd.isna(raw_ph) or raw_ph == '' else (float(raw_ph), 1)
        raw_temp = row.get('Temperature')
        temp_val, use_temp = (0.0, 0) if pd.isna(raw_temp) or raw_temp == '' else (float(raw_temp), 1)

        def get_val_mask(val):
            if pd.isna(val): return 0.0, 0.0
            return float(val), 1.0

        vkcat, mkcat = get_val_mask(row.get('log10_kcat'))
        vkm, mkm = get_val_mask(row.get('log10_km'))
        vki, mki = get_val_mask(row.get('log10_ki'))

        labels = torch.tensor([vkcat, vkm, vki], dtype=torch.float32)
        masks = torch.tensor([mkcat, mkm, mki], dtype=torch.float32)

        return prot_tensor, mol_tensor, (ph_val, use_ph), (temp_val, use_temp), labels, masks



def main():
    prot_feat_path, mol_feat_path = r"D:\data\prot_embedding.h5", r"D:\data\mol_embedding_un.h5"
    tasks = ['kcat', 'km', 'ki']

    #
    TARGET_R2 = {
        'kcat': 0.6603,
        'km': 0.6237,
        'ki': 0.6086
    }

    # 初始化上一轮的 R2 记录
    last_epoch_r2 = {t: 0.01 for t in tasks}

    # 实例化数据集 (默认从 data_dir 读取 train.csv, val.csv, test.csv)
    train_ds = UnifiedEnzymeDataset(os.path.join(args.data_dir, 'train.csv'), prot_feat_path, mol_feat_path)
    val_ds = UnifiedEnzymeDataset(os.path.join(args.data_dir, 'val.csv'), prot_feat_path, mol_feat_path)
    test_ds = UnifiedEnzymeDataset(os.path.join(args.data_dir, 'test.csv'), prot_feat_path, mol_feat_path)

    # 标准 DataLoader 加载，彻底去除采样权重
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=unified_collate,
                          pin_memory=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=unified_collate,
                        pin_memory=True, num_workers=args.num_workers)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=unified_collate,
                         pin_memory=True, num_workers=args.num_workers)

    # 独立初始化各任务的 ASER 不平衡损失函数
    task_loss_fns = {}
    kcat_targets = train_ds.df['log10_kcat'].dropna().values
    km_targets = train_ds.df['log10_km'].dropna().values
    ki_targets = train_ds.df['log10_ki'].dropna().values
    task_loss_fns['kcat'] = ASERLoss(all_targets_of_train_set=kcat_targets, k=1)
    task_loss_fns['km'] = ASERLoss(all_targets_of_train_set=km_targets, k=1)
    task_loss_fns['ki'] = ASERLoss(all_targets_of_train_set=ki_targets, k=1)

    model = Enzyme(hidden_dim=args.hidden_dim, expert_total=10, device=device).to(device)

    # AMTL 聚焦超参数
    amtl_gamma = 2.0
    margin=1.05

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, capturable=True)
    current_lr = args.lr

    file_path = f'log/[{args.model}]_MGMI_AMTL_head' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.txt'
    best_epoch, best_valid_res, no_improve_counter = 0, float('inf'), 0
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('log', exist_ok=True)
    checkpoint_path = f'checkpoints/resume_{args.model}_MGMI_AMTL_head.pt'
    start_epoch = 1
    # -------------------- 恢复断点 --------------------
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


        for param_group in optimizer.param_groups:
            param_group['capturable'] = True

        start_epoch = checkpoint['epoch'] + 1
        current_lr = checkpoint.get('current_lr', current_lr)
        # current_lr = checkpoint.get('current_lr', current_lr)
        best_epoch, best_valid_res = checkpoint['best_epoch'], checkpoint['best_valid_res']
        no_improve_counter = checkpoint.get('no_improve_counter', 0)
        # 恢复 R2 记录
        last_epoch_r2 = checkpoint.get('last_epoch_r2', last_epoch_r2)
        print(f"成功恢复！将从第 {start_epoch} 轮继续训练...")

    # -------------------- Training Loop --------------------
    for epoch in tqdm(range(start_epoch, args.epochs + 1)):
        model.train()
        train_loss_sum = {task: 0.0 for task in tasks}
        valid_step_counts = {task: 0 for task in tasks}
        optimizer.param_groups[0]['lr'] = current_lr

        for batch_data in tqdm(train_dl, position=0, ncols=150, colour='#666666', desc=f'Epoch {epoch} Training'):
            optimizer.zero_grad()

            prot_emb = batch_data['protein_emb'].to(device)
            prot_mask = batch_data['protein_mask'].to(device)
            mol_emb = batch_data['mol_emb'].to(device)
            mol_mask = batch_data['mol_mask'].to(device)
            ph = batch_data['ph'].to(device)
            use_ph = batch_data['use_ph'].to(device)
            temp = batch_data['temp'].to(device)
            use_temp = batch_data['use_temp'].to(device)
            labels = batch_data['labels'].to(device)  # [B, 3]
            masks = batch_data['masks'].to(device)  # [B, 3]

            # MGMI 多模态多任务前向传播
            task_outputs, ortho_loss, moe_loss = model(
                prot_emb, prot_mask, mol_emb, mol_mask, ph, use_ph, temp, use_temp
            )

            batch_task_losses = {}
            valid_tasks = []

            # 过滤有效标签并计算各任务 ASER 损失
            for i, task in enumerate(tasks):
                mask_t = masks[:, i].bool()
                if mask_t.sum() > 0:
                    y_pred_t = task_outputs[task].flatten()[mask_t]
                    y_true_t = labels[:, i][mask_t]

                    loss_t = task_loss_fns[task](y_pred_t, y_true_t)
                    batch_task_losses[task] = loss_t
                    valid_tasks.append(task)

                    with torch.no_grad():
                        train_loss_sum[task] += F.mse_loss(y_pred_t, y_true_t).item()
                        valid_step_counts[task] += 1

            step_loss = torch.tensor(0.0, device=device)
            softmax_weights = []

            # 2. AMTL: 基于 R2 达成度的加权求和
            if valid_tasks:
                achievements_weights = []
                for task in valid_tasks:

                    current_r2 = max(0.001, last_epoch_r2[task])
                    target_r2 = TARGET_R2[task]

                    # 计算达成度 (Achievement): 限制在 [0, 1]

                    achievement = torch.tensor(current_r2 / (margin*target_r2), dtype=torch.float32, device=device)
                    achievement = torch.clamp(achievement, 0.0, 1.0)

                    # 计算焦点权重
                    v_t = (1.0 - achievement) ** amtl_gamma
                    achievements_weights.append(v_t)

                # 权重 Softmax 归一化
                weights_tensor = torch.stack(achievements_weights)
                softmax_weights = F.softmax(weights_tensor, dim=0) * len(valid_tasks)


                for idx, task in enumerate(valid_tasks):
                    step_loss += softmax_weights[idx] * batch_task_losses[task]


            final_loss = step_loss + ortho_loss * 0.5 + moe_loss
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # -------------------- 日志记录 --------------------
        with open(file_path, "a", encoding='utf-8') as f:
            f.write(f'\nEpoch: {epoch} | lr: {current_lr}\n')

            if valid_tasks:
                weight_str = ", ".join(
                    [f"{t}: {w:.4f}" for t, w in zip(valid_tasks, softmax_weights.detach().cpu().numpy())])
                # print(f"AMTL  Weights: {weight_str}")
                f.write(f'AMTL  Weights: {weight_str}\n')

            for task in tasks:
                if valid_step_counts[task] > 0:
                    avg_train_loss = train_loss_sum[task] / valid_step_counts[task]
                    print(f'Train loss (MSE) [{task}]: {avg_train_loss:.4f}')
                    f.write(f'Train loss (MSE) [{task}]: {avg_train_loss:.4f}\n')

        # -------------------- 验证阶段 --------------------
        with torch.no_grad():
            model.eval()
            current_epoch_rmse_sum = 0

            all_preds = {t: [] for t in tasks}
            all_labels = {t: [] for t in tasks}

            for batch_data in val_dl:
                prot_emb, prot_mask = batch_data['protein_emb'].to(device), batch_data['protein_mask'].to(device)
                mol_emb, mol_mask = batch_data['mol_emb'].to(device), batch_data['mol_mask'].to(device)
                ph, use_ph = batch_data['ph'].to(device), batch_data['use_ph'].to(device)
                temp, use_temp = batch_data['temp'].to(device), batch_data['use_temp'].to(device)
                labels = batch_data['labels'].to(device)
                masks = batch_data['masks'].to(device)

                task_outputs, _, _ = model(prot_emb, prot_mask, mol_emb, mol_mask, ph, use_ph, temp, use_temp)

                for i, task in enumerate(tasks):
                    mask_t = masks[:, i].bool()
                    if mask_t.sum() > 0:
                        all_preds[task].append(task_outputs[task].flatten()[mask_t].detach().cpu().numpy())
                        all_labels[task].append(labels[:, i][mask_t].detach().cpu().numpy())

            for task in tasks:
                if len(all_preds[task]) > 0:
                    t_preds = np.concatenate(all_preds[task], axis=0)
                    t_labels = np.concatenate(all_labels[task], axis=0)

                    val_rmse = np.sqrt(mean_squared_error(t_labels, t_preds))
                    val_r2 = r2_score(t_labels, t_preds)
                    val_mae = mean_absolute_error(t_labels, t_preds)
                    val_pcc = pearsonr(t_labels.flatten(), t_preds.flatten())[0] if len(t_preds) > 1 else 0.0

                    # 更新上一轮 R2 记录，供下一 Epoch 的 AMTL 进度平衡使用
                    last_epoch_r2[task] = val_r2

                    current_epoch_rmse_sum += val_rmse
                    print(
                        f'Valid [{task}] RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | R2: {val_r2:.4f} | PCC: {val_pcc:.4f}')
                    with open(file_path, "a", encoding='utf-8') as f:
                        f.write(
                            f'Valid [{task}] RMSE:{val_rmse:.4f} MAE:{val_mae:.4f} R2:{val_r2:.4f} PCC:{val_pcc:.4f}\n')

            # 依据三任务的总 RMSE 保存最佳模型
            if current_epoch_rmse_sum < best_valid_res:
                best_epoch, best_valid_res = epoch, current_epoch_rmse_sum
                torch.save(model.state_dict(), 'checkpoints/' + file_path.split('/')[-1][:-4] + '.pt')
                print(f"New Best Model Saved! (Sum RMSE: {best_valid_res:.4f})")
                no_improve_counter = 0
            else:
                no_improve_counter += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'current_lr': current_lr,
            'best_epoch': best_epoch,
            'best_valid_res': best_valid_res,
            'no_improve_counter': no_improve_counter,
            'last_epoch_r2': last_epoch_r2  #
        }, checkpoint_path)

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if no_improve_counter > early_stop:
            print(f"Early stopping triggered after {early_stop} epochs.")
            break


    print("\nStarting Testing with Best Model...")
    with torch.no_grad():
        model_path = 'checkpoints/' + file_path.split('/')[-1][:-4] + '.pt'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        all_preds = {t: [] for t in tasks}
        all_labels = {t: [] for t in tasks}

        for batch_data in tqdm(test_dl, desc='Testing'):
            prot_emb, prot_mask = batch_data['protein_emb'].to(device), batch_data['protein_mask'].to(device)
            mol_emb, mol_mask = batch_data['mol_emb'].to(device), batch_data['mol_mask'].to(device)
            ph, use_ph = batch_data['ph'].to(device), batch_data['use_ph'].to(device)
            temp, use_temp = batch_data['temp'].to(device), batch_data['use_temp'].to(device)
            labels = batch_data['labels'].to(device)
            masks = batch_data['masks'].to(device)

            task_outputs, _, _ = model(prot_emb, prot_mask, mol_emb, mol_mask, ph, use_ph, temp, use_temp)

            for i, task in enumerate(tasks):
                mask_t = masks[:, i].bool()
                if mask_t.sum() > 0:
                    all_preds[task].append(task_outputs[task].flatten()[mask_t].detach().cpu().numpy())
                    all_labels[task].append(labels[:, i][mask_t].detach().cpu().numpy())

        for task in tasks:
            if len(all_preds[task]) > 0:
                t_preds = np.concatenate(all_preds[task], axis=0)
                t_labels = np.concatenate(all_labels[task], axis=0)

                test_rmse = np.sqrt(mean_squared_error(t_labels, t_preds))
                test_r2 = r2_score(t_labels, t_preds)
                test_mae = mean_absolute_error(t_labels, t_preds)
                test_pcc = pearsonr(t_labels.flatten(), t_preds.flatten())[0] if len(t_preds) > 1 else 0.0

                print(
                    f'Test Result - Task: {task} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R2: {test_r2:.4f} | PCC: {test_pcc:.4f}')
                with open(file_path, "a", encoding='utf-8') as f:
                    f.write(
                        f'Task: {task} Test RMSE:{test_rmse:.4f} MAE:{test_mae:.4f} R2:{test_r2:.4f} PCC:{test_pcc:.4f}\n')


if __name__ == '__main__':
    main()
