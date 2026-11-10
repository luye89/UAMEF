import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
from model.module import MoE  # 确保你的环境里有这个模块

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def calculate_ortho_loss(input_vec):
    """
    计算正交损失 (Orthogonal Loss)。
    通过惩罚特征协方差矩阵的非对角线元素，促使不同模态提取到的特征相互独立（去相关），
    从而减少冗余信息，提高多模态表征的丰富度。
    """
    x = input_vec - torch.mean(input_vec, axis=2).unsqueeze(2).repeat(1, 1, input_vec.shape[2])
    cov_matrix = torch.matmul(x, x.transpose(1, 2)) / (x.shape[2] - 1)
    loss = (torch.sum(cov_matrix ** 2) - torch.sum(torch.diagonal(cov_matrix, dim1=1, dim2=2) ** 2)) / (
            cov_matrix.shape[0] * (cov_matrix.shape[1] - 1) * (cov_matrix.shape[2] - 1) + 1e-8)
    return loss


def temperature_scaled_softmax(logits, temperature=1.0, dim=0):
    """
    带温度缩放的 Softmax，
    用于平滑或锐化最终模态融合的权重分布。
    """
    logits = logits / temperature
    return torch.softmax(logits, dim=dim)


class GaussianSmearing(nn.Module):
    """
    高斯涂抹 (Gaussian Smearing) 模块。
    """
    def __init__(self, start=0.0, stop=1.0, num_gaussians=64):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class GlobalMAG(nn.Module):
    """
    全局模态自适应门控模块 (Modality-Adaptive Gating, MAG)。
    """
    def __init__(self, hidden_dim, num_modalities=4, gate_th=0.1):
        super(GlobalMAG, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.gate_th = gate_th

    def forward(self, features_list, valid_masks):
        g_scores = []
        for x in features_list:
            pooled = x.mean(dim=1)
            h = F.relu(self.fc1(pooled))
            g = torch.sigmoid(self.fc2(h))
            g_scores.append(g)

        g_tensor = torch.cat(g_scores, dim=1)
        mask = (g_tensor >= self.gate_th) & valid_masks.bool()
        g_masked = g_tensor.masked_fill(~mask, -1e9)
        alphas = F.softmax(g_masked, dim=1)

        out_features = []
        for i, x in enumerate(features_list):
            alpha_i = alphas[:, i].view(-1, 1, 1)
            out_features.append(x * alpha_i)

        return out_features, alphas


class InterIntraFusionBlock(nn.Module):
    """
    模态内与模态间特征融合块。
    """
    def __init__(self, hidden_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, target_feat, source_feats, key_padding_mask=None):
        cross_out, _ = self.cross_attn(query=target_feat, key=source_feats, value=source_feats, key_padding_mask=key_padding_mask)
        x = self.norm1(target_feat + cross_out)

        self_out, _ = self.self_attn(query=x, key=x, value=x)
        x = self.norm2(x + self_out)

        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x


class Enzyme(nn.Module):
    def __init__(self, prot_dim=1024, mol_dim=300, rbf_dim=64, hidden_dim=128,
                 batch_first=True, dropout=0.1, layers=4, expert_k=2, expert_total=10, device=torch.device('cuda')):
        super(Enzyme, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim

        # =========================================================
        # 模态特征初始化层
        # =========================================================
        self.prot_projection = nn.Linear(prot_dim, hidden_dim)
        self.prot_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.prot_pos_embed = nn.Parameter(torch.zeros(1, 1024, hidden_dim))

        self.mol_projection = nn.Linear(mol_dim, hidden_dim)
        self.mol_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # =========================================================
        # pH / Temperature RBF
        # =========================================================
        self.rbf_ph = GaussianSmearing(start=0.0, stop=14.0, num_gaussians=rbf_dim)
        self.rbf_temp = GaussianSmearing(start=0.0, stop=100.0, num_gaussians=rbf_dim)
        self.ph_projection = nn.Linear(rbf_dim, hidden_dim)
        self.temp_projection = nn.Linear(rbf_dim, hidden_dim)

        # =========================================================
        # 深度多层融合网络 (GlobalMAG + Inter/Intra Fusion)
        # =========================================================
        self.fusion_layers = nn.ModuleList([
            nn.ModuleDict({
                'mag': GlobalMAG(hidden_dim, num_modalities=4),
                'prot': InterIntraFusionBlock(hidden_dim, num_heads=2, dropout=dropout),
                'mol': InterIntraFusionBlock(hidden_dim, num_heads=2, dropout=dropout),
                'ph': InterIntraFusionBlock(hidden_dim, num_heads=2, dropout=dropout),
                'temp': InterIntraFusionBlock(hidden_dim, num_heads=2, dropout=dropout)
            }) for _ in range(layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # =========================================================
        # 混合专家系统 MoE
        # =========================================================
        self.moe = MoE(hidden_dim, hidden_dim, hidden_dim, expert_total, hidden_dim, noisy_gating=True, k=expert_k)

        # =========================================================
        # [修改点] Attention Branch: 使用 Self-Attention 并拼接降维
        # =========================================================
        # 删除了原来的 self.fusion_query 探针
        self.final_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.attn_fusion = nn.Linear(hidden_dim * 4, hidden_dim)

        # =========================================================
        # Gate Branch
        # =========================================================
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 最终归一化
        self.mm_layernorm = nn.LayerNorm(hidden_dim)

        # =========================================================
        # ★ 独立多任务预测头 (为适配你的 AMTL 多任务结构保留)
        # =========================================================
        self.head_kcat = nn.Linear(hidden_dim, 1)
        self.head_km = nn.Linear(hidden_dim, 1)
        self.head_ki = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """网络参数的 Xavier 初始化策略"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, prot_feat, prot_mask, mol_feat, mol_mask, ph_feat, use_ph, temp_feat, use_temp):
        batch_size = prot_feat.shape[0]

        # Protein
        prot_embed = self.prot_projection(prot_feat)
        prot_embed = prot_embed + self.prot_pos_embed[:, :prot_embed.shape[1], :]
        prot_embed = torch.cat((self.prot_cls_token.repeat(batch_size, 1, 1), prot_embed), dim=1)
        p_mask = torch.cat((torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device), (prot_mask == 0)), dim=1)

        # Molecule
        mol_embed = self.mol_projection(mol_feat)
        mol_embed = torch.cat((self.mol_cls_token.repeat(batch_size, 1, 1), mol_embed), dim=1)
        m_mask = torch.cat((torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device), (mol_mask == 0)), dim=1)

        # pH
        ph_embed = self.ph_projection(self.rbf_ph(ph_feat)).unsqueeze(1) if use_ph.sum() != 0 else torch.zeros(batch_size, 1, self.hidden_dim).to(self.device)

        # Temperature
        temp_embed = self.temp_projection(self.rbf_temp(temp_feat)).unsqueeze(1) if use_temp.sum() != 0 else torch.zeros(batch_size, 1, self.hidden_dim).to(self.device)

        ph_mask = (use_ph == 0).bool()
        temp_mask = (use_temp == 0).bool()

        # Input LayerNorm
        p_emb = self.input_norm(prot_embed)
        m_emb = self.input_norm(mol_embed)
        ph_emb = self.input_norm(ph_embed)
        temp_emb = self.input_norm(temp_embed)

        # 模态有效状态
        valid_masks = torch.cat([
            torch.ones(batch_size, 1, device=self.device),
            torch.ones(batch_size, 1, device=self.device),
            use_ph.float(),
            use_temp.float()
        ], dim=1)

        final_alphas = None

        # =========================================================
        # GlobalMAG + Inter/Intra Fusion
        # =========================================================
        for layer in self.fusion_layers:
            scaled_feats, alphas = layer['mag']([p_emb, m_emb, ph_emb, temp_emb], valid_masks)
            sp_emb, sm_emb, sph_emb, stemp_emb = scaled_feats
            final_alphas = alphas

            # Protein <- other modalities
            src_prot = torch.cat([sm_emb, sph_emb, stemp_emb], dim=1)
            mask_prot = torch.cat([m_mask, ph_mask, temp_mask], dim=1)
            new_p = layer['prot'](p_emb, src_prot, key_padding_mask=mask_prot)

            # Molecule <- other modalities
            src_mol = torch.cat([sp_emb, sph_emb, stemp_emb], dim=1)
            mask_mol = torch.cat([p_mask, ph_mask, temp_mask], dim=1)
            new_m = layer['mol'](m_emb, src_mol, key_padding_mask=mask_mol)

            # pH <- other modalities
            src_ph = torch.cat([sp_emb, sm_emb, stemp_emb], dim=1)
            mask_ph = torch.cat([p_mask, m_mask, temp_mask], dim=1)
            new_ph = layer['ph'](ph_emb, src_ph, key_padding_mask=mask_ph)

            # Temperature <- other modalities
            src_temp = torch.cat([sp_emb, sm_emb, sph_emb], dim=1)
            mask_temp = torch.cat([p_mask, m_mask, ph_mask], dim=1)
            new_temp = layer['temp'](temp_emb, src_temp, key_padding_mask=mask_temp)

            p_emb, m_emb, ph_emb, temp_emb = new_p, new_m, new_ph, new_temp

        p_emb = self.dropout(p_emb)
        m_emb = self.dropout(m_emb)

        prot_cls_feat = p_emb[:, 0].unsqueeze(1) * final_alphas[:, 0].view(-1, 1, 1)
        mol_cls_feat = m_emb[:, 0].unsqueeze(1) * final_alphas[:, 1].view(-1, 1, 1)
        ph_token = ph_emb * final_alphas[:, 2].view(-1, 1, 1)
        temp_token = temp_emb * final_alphas[:, 3].view(-1, 1, 1)

        mm_embed = torch.cat((prot_cls_feat, mol_cls_feat, ph_token, temp_token), dim=1)
        mm_embed = self.dropout(mm_embed)

        # =========================================================
        # MoE Mask & Execution
        # =========================================================
        mm_mask = torch.zeros(batch_size, 4).to(self.device)
        mm_mask[:, 2] = (1 - use_ph).squeeze(1).float()
        mm_mask[:, 3] = (1 - use_temp).squeeze(1).float()

        mm_moe = torch.zeros(batch_size * 4, self.hidden_dim).to(self.device)
        valid_indices = (mm_mask.reshape(-1) == 0)

        if valid_indices.sum() > 0:
            input_feat = mm_embed.reshape(-1, self.hidden_dim)[valid_indices]
            tmp_moe, moe_loss = self.moe(input_feat, input_feat)
            mm_moe[valid_indices] = tmp_moe
        else:
            moe_loss = torch.tensor(0.0).to(self.device)

        mm_moe = mm_moe.reshape(batch_size, 4, self.hidden_dim)

     
        attn_out, _ = self.final_attn(
            query=mm_moe,
            key=mm_moe,
            value=mm_moe,
            key_padding_mask=mm_mask.bool(),  # 忽略缺失模态
            need_weights=False
        )

        # 缺失模态输出置零
        attn_out = attn_out.masked_fill(mm_mask.bool().unsqueeze(-1), 0.0)

        # [B, 4, 128] → [B, 512]
        attn_concat = attn_out.reshape(batch_size, 4 * self.hidden_dim)

        # [B, 512] → [B, 128]
        attn_feat = self.attn_fusion(attn_concat)

    
        gate = torch.sigmoid(self.gate_net(mm_moe))
        gate = gate * (1 - mm_mask.unsqueeze(2).float()) 

        gated_feat = (gate * mm_moe).sum(dim=1)
        gate_norm = gate.sum(dim=1) + 1e-8
        gated_feat = gated_feat / gate_norm

     
        valid_count = (1 - mm_mask.float()).sum(dim=1).unsqueeze(1) + 1e-8
        sum_feat = (mm_moe * (1 - mm_mask.unsqueeze(2).float())).sum(dim=1)
        mean_feat = sum_feat / valid_count

    
        weighted_mm = attn_feat + gated_feat + mean_feat
        final_mm_embed = self.dropout(self.mm_layernorm(weighted_mm))

      
        task_outputs = {
            'kcat': self.head_kcat(final_mm_embed),
            'km': self.head_km(final_mm_embed),
            'ki': self.head_ki(final_mm_embed)
        }

        ortho_loss = calculate_ortho_loss(mm_embed)

        return task_outputs, ortho_loss, moe_loss
