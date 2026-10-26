# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# import random
# import numpy as np
# from model.module import MoE  # 确保你的环境里有这个模块
#
# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
#
#
# def calculate_ortho_loss(input_vec):
#     x = input_vec - torch.mean(input_vec, axis=2).unsqueeze(2).repeat(1, 1, input_vec.shape[2])
#     cov_matrix = torch.matmul(x, x.transpose(1, 2)) / (x.shape[2] - 1)
#     loss = (torch.sum(cov_matrix ** 2) - torch.sum(torch.diagonal(cov_matrix, dim1=1, dim2=2) ** 2)) / (
#             cov_matrix.shape[0] * (cov_matrix.shape[1] - 1) * (cov_matrix.shape[2] - 1) + 1e-8)
#     return loss
#
#
# class GaussianSmearing(nn.Module):
#     def __init__(self, start=0.0, stop=1.0, num_gaussians=64):
#         super(GaussianSmearing, self).__init__()
#         offset = torch.linspace(start, stop, num_gaussians)
#         self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
#         self.register_buffer('offset', offset)
#
#     def forward(self, dist):
#         dist = dist.view(-1, 1) - self.offset.view(1, -1)
#         return torch.exp(self.coeff * torch.pow(dist, 2))
#
#
# class GlobalMAG(nn.Module):
#     def __init__(self, hidden_dim, num_modalities=4, gate_th=0.1):
#         super(GlobalMAG, self).__init__()
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)
#         self.gate_th = gate_th
#
#     def forward(self, features_list, valid_masks):
#         g_scores = []
#         for x in features_list:
#             pooled = x.mean(dim=1)
#             h = F.relu(self.fc1(pooled))
#             g = torch.sigmoid(self.fc2(h))
#             g_scores.append(g)
#
#         g_tensor = torch.cat(g_scores, dim=1)
#         mask = (g_tensor >= self.gate_th) & valid_masks.bool()
#         g_masked = g_tensor.masked_fill(~mask, -1e9)
#         alphas = F.softmax(g_masked, dim=1)
#
#         out_features = []
#         for i, x in enumerate(features_list):
#             alpha_i = alphas[:, i].view(-1, 1, 1)
#             out_features.append(x * alpha_i)
#
#         return out_features, alphas
#
#
# class InterIntraFusionBlock(nn.Module):
#     def __init__(self, hidden_dim, num_heads=2, dropout=0.1):
#         super().__init__()
#         self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
#         self.norm1 = nn.LayerNorm(hidden_dim)
#
#         self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 4, hidden_dim)
#         )
#         self.norm3 = nn.LayerNorm(hidden_dim)
#
#     def forward(self, target_feat, source_feats, key_padding_mask=None):
#         cross_out, _ = self.cross_attn(query=target_feat, key=source_feats, value=source_feats,
#                                        key_padding_mask=key_padding_mask)
#         x = self.norm1(target_feat + cross_out)
#
#         self_out, _ = self.self_attn(query=x, key=x, value=x)
#         x = self.norm2(x + self_out)
#
#         ffn_out = self.ffn(x)
#         x = self.norm3(x + ffn_out)
#         return x
#
#
# class Enzyme(nn.Module):
#     def __init__(self, prot_dim=1024, mol_dim=300, rbf_dim=64, hidden_dim=128,
#                  batch_first=True, dropout=0.1, layers=4, expert_k=2, expert_total=10, device=torch.device('cuda')):
#         super(Enzyme, self).__init__()
#         self.device = device
#         self.hidden_dim = hidden_dim
#
#         # --- 模态特征初始化层 ---
#         self.prot_projection = nn.Linear(prot_dim, hidden_dim)
#         self.prot_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
#         self.prot_pos_embed = nn.Parameter(torch.zeros(1, 1024, hidden_dim))
#
#         self.mol_projection = nn.Linear(mol_dim, hidden_dim)
#         self.mol_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
#
#         self.rbf_ph = GaussianSmearing(start=0.0, stop=14.0, num_gaussians=rbf_dim)
#         self.rbf_temp = GaussianSmearing(start=0.0, stop=100.0, num_gaussians=rbf_dim)
#         self.ph_projection = nn.Linear(rbf_dim, hidden_dim)
#         self.temp_projection = nn.Linear(rbf_dim, hidden_dim)
#
#         # --- 深度多层融合网络 ---
#         self.fusion_layers = nn.ModuleList([
#             nn.ModuleDict({
#                 'mag': GlobalMAG(hidden_dim, num_modalities=4),
#                 'prot': InterIntraFusionBlock(hidden_dim, num_heads=2, dropout=dropout),
#                 'mol': InterIntraFusionBlock(hidden_dim, num_heads=2, dropout=dropout),
#                 'ph': InterIntraFusionBlock(hidden_dim, num_heads=2, dropout=dropout),
#                 'temp': InterIntraFusionBlock(hidden_dim, num_heads=2, dropout=dropout)
#             }) for _ in range(layers)
#         ])
#
#         self.dropout = nn.Dropout(dropout)
#         self.input_norm = nn.LayerNorm(hidden_dim)
#
#         # --- 混合专家系统 (MoE) ---
#         self.moe = MoE(hidden_dim, hidden_dim, hidden_dim, expert_total, hidden_dim, noisy_gating=True, k=expert_k)
#
#         # =========================================================
#         # 1. 横向拼接自注意力 (Attention Branch)
#         # =========================================================
#         self.final_attn = nn.MultiheadAttention(
#             embed_dim=hidden_dim * 4,  # 128 * 4 = 512
#             num_heads=4,
#             dropout=0.0,  # 避免唯一的注意力权重被丢弃
#             batch_first=True
#         )
#         # 将512维融合特征压缩到128维
#         self.attn_fusion = nn.Linear(hidden_dim * 4, hidden_dim)
#
#         # 2. 特征级门控网络 (Gate Branch)
#         self.gate_net = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
#
#         # 3. 最终归一化与单任务预测头
#         self.mm_layernorm = nn.LayerNorm(hidden_dim)
#         self.head_pred = nn.Linear(hidden_dim, 1)
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             nn.init.xavier_uniform_(module.weight)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.LayerNorm):
#             nn.init.ones_(module.weight)
#             nn.init.zeros_(module.bias)
#
#     def forward(self, prot_feat, prot_mask, mol_feat, mol_mask, ph_feat, use_ph, temp_feat, use_temp):
#         batch_size = prot_feat.shape[0]
#
#         prot_embed = self.prot_projection(prot_feat)
#         prot_embed = prot_embed + self.prot_pos_embed[:, :prot_embed.shape[1], :]
#         prot_embed = torch.cat((self.prot_cls_token.repeat(batch_size, 1, 1), prot_embed), dim=1)
#         p_mask = torch.cat((torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device), (prot_mask == 0)), dim=1)
#
#         mol_embed = self.mol_projection(mol_feat)
#         mol_embed = torch.cat((self.mol_cls_token.repeat(batch_size, 1, 1), mol_embed), dim=1)
#         m_mask = torch.cat((torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device), (mol_mask == 0)), dim=1)
#
#         ph_embed = self.ph_projection(self.rbf_ph(ph_feat)).unsqueeze(1) if use_ph.sum() != 0 else torch.zeros(
#             batch_size, 1, self.hidden_dim).to(self.device)
#         temp_embed = self.temp_projection(self.rbf_temp(temp_feat)).unsqueeze(
#             1) if use_temp.sum() != 0 else torch.zeros(batch_size, 1, self.hidden_dim).to(self.device)
#
#         ph_mask = (use_ph == 0).bool()
#         temp_mask = (use_temp == 0).bool()
#
#         p_emb, m_emb = self.input_norm(prot_embed), self.input_norm(mol_embed)
#         ph_emb, temp_emb = self.input_norm(ph_embed), self.input_norm(temp_embed)
#
#         valid_masks = torch.cat([
#             torch.ones(batch_size, 1, device=self.device),
#             torch.ones(batch_size, 1, device=self.device),
#             use_ph.float(),
#             use_temp.float()
#         ], dim=1)
#
#         final_alphas = None
#
#         for layer in self.fusion_layers:
#             scaled_feats, alphas = layer['mag']([p_emb, m_emb, ph_emb, temp_emb], valid_masks)
#             sp_emb, sm_emb, sph_emb, stemp_emb = scaled_feats
#             final_alphas = alphas
#
#             src_prot = torch.cat([sm_emb, sph_emb, stemp_emb], dim=1)
#             mask_prot = torch.cat([m_mask, ph_mask, temp_mask], dim=1)
#             new_p = layer['prot'](p_emb, src_prot, key_padding_mask=mask_prot)
#
#             src_mol = torch.cat([sp_emb, sph_emb, stemp_emb], dim=1)
#             mask_mol = torch.cat([p_mask, ph_mask, temp_mask], dim=1)
#             new_m = layer['mol'](m_emb, src_mol, key_padding_mask=mask_mol)
#
#             src_ph = torch.cat([sp_emb, sm_emb, stemp_emb], dim=1)
#             mask_ph = torch.cat([p_mask, m_mask, temp_mask], dim=1)
#             new_ph = layer['ph'](ph_emb, src_ph, key_padding_mask=mask_ph)
#
#             src_temp = torch.cat([sp_emb, sm_emb, sph_emb], dim=1)
#             mask_temp = torch.cat([p_mask, m_mask, ph_mask], dim=1)
#             new_temp = layer['temp'](temp_emb, src_temp, key_padding_mask=mask_temp)
#
#             p_emb, m_emb, ph_emb, temp_emb = new_p, new_m, new_ph, new_temp
#
#         p_emb, m_emb = self.dropout(p_emb), self.dropout(m_emb)
#
#         prot_cls_feat = p_emb[:, 0].unsqueeze(1) * final_alphas[:, 0].view(-1, 1, 1)
#         mol_cls_feat = m_emb[:, 0].unsqueeze(1) * final_alphas[:, 1].view(-1, 1, 1)
#         ph_token = ph_emb * final_alphas[:, 2].view(-1, 1, 1)
#         temp_token = temp_emb * final_alphas[:, 3].view(-1, 1, 1)
#
#         mm_embed = torch.cat((prot_cls_feat, mol_cls_feat, ph_token, temp_token), dim=1)
#         mm_embed = self.dropout(mm_embed)
#
#         mm_mask = torch.zeros(batch_size, 4).to(self.device)
#         mm_mask[:, 2] = (1 - use_ph).squeeze(1).float()
#         mm_mask[:, 3] = (1 - use_temp).squeeze(1).float()
#
#         mm_moe = torch.zeros(batch_size * 4, self.hidden_dim).to(self.device)
#         valid_indices = (mm_mask.reshape(-1) == 0)
#
#         if valid_indices.sum() > 0:
#             input_feat = mm_embed.reshape(-1, self.hidden_dim)[valid_indices]
#             tmp_moe, moe_loss = self.moe(input_feat, input_feat)
#             mm_moe[valid_indices] = tmp_moe
#         else:
#             moe_loss = torch.tensor(0.0).to(self.device)
#
#         mm_moe = mm_moe.reshape(batch_size, 4, self.hidden_dim)
#
#         # =====================================================================
#         #  [Branch 1] Horizontal Self-Attention Fusion (横向拼接自注意力)
#         # =====================================================================
#         # 1. 缺失模态置零
#         h = mm_moe.masked_fill(mm_mask.bool().unsqueeze(-1), 0.0)
#
#         # 2. 横向拼接 [B, 4, 128] -> [B, 1, 512]
#         h_concat = h.reshape(batch_size, 1, 4 * self.hidden_dim)
#
#         # 3. 联合特征自注意力
#         attn_out, _ = self.final_attn(
#             query=h_concat,
#             key=h_concat,
#             value=h_concat,
#             need_weights=False
#         )
#
#         # 4. 降维 [B, 1, 512] -> [B, 128]
#         attn_feat = self.attn_fusion(attn_out.squeeze(1))
#
#         # =====================================================================
#         #  [Branch 2] Gate Branch (特征级精细过滤，去噪解耦)
#         # =====================================================================
#         gate = torch.sigmoid(self.gate_net(mm_moe))
#         gate = gate * (1 - mm_mask.unsqueeze(2).float())
#
#         gated_feat = (gate * mm_moe).sum(dim=1)
#         gate_norm = gate.sum(dim=1) + 1e-8
#         gated_feat = gated_feat / gate_norm
#
#         # =====================================================================
#         #  [Branch 3] Mean Branch (基础特征均衡混合)
#         # =====================================================================
#         valid_count = (1 - mm_mask.float()).sum(dim=1).unsqueeze(1) + 1e-8
#         sum_feat = (mm_moe * (1 - mm_mask.unsqueeze(2).float())).sum(dim=1)
#         mean_feat = sum_feat / valid_count
#
#         # =====================================================================
#         #  三路混合相加
#         # =====================================================================
#         weighted_mm = attn_feat + gated_feat + mean_feat
#         final_mm_embed = self.dropout(self.mm_layernorm(weighted_mm))
#
#         # 最终打分与去相关损失计算
#         scores = self.head_pred(final_mm_embed)
#         ortho_loss = calculate_ortho_loss(mm_embed)
#
#         return scores, ortho_loss, moe_loss
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
    """带温度缩放的 Softmax，用于平滑或锐化最终模态融合的权重分布。"""
    logits = logits / temperature
    return torch.softmax(logits, dim=dim)


class GaussianSmearing(nn.Module):
    """
    高斯涂抹 (Gaussian Smearing) 模块。
    将一维的连续标量（如 pH 值、温度）映射到高维的高斯径向基函数 (RBF) 空间中，
    增强模型对连续环境条件的表达能力。
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
    基于论文公式 (1), (2), (3), (4) 实现，动态评估各模态数据质量并分配全局相对权重。
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
        cross_out, _ = self.cross_attn(query=target_feat, key=source_feats, value=source_feats,
                                       key_padding_mask=key_padding_mask)
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

        # --- 模态特征初始化层 ---
        self.prot_projection = nn.Linear(prot_dim, hidden_dim)
        self.prot_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.prot_pos_embed = nn.Parameter(torch.zeros(1, 1024, hidden_dim))

        self.mol_projection = nn.Linear(mol_dim, hidden_dim)
        self.mol_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.rbf_ph = GaussianSmearing(start=0.0, stop=14.0, num_gaussians=rbf_dim)
        self.rbf_temp = GaussianSmearing(start=0.0, stop=100.0, num_gaussians=rbf_dim)
        self.ph_projection = nn.Linear(rbf_dim, hidden_dim)
        self.temp_projection = nn.Linear(rbf_dim, hidden_dim)

        # --- 深度多层融合网络 ---
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

        # --- 混合专家系统 (MoE) ---
        self.moe = MoE(hidden_dim, hidden_dim, hidden_dim, expert_total, hidden_dim, noisy_gating=True, k=expert_k)

        # 1. 模态交互自注意力 (用于 Attention Branch)
        self.final_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)

        # 2. 自注意力特征聚合层 (将 4 个模态的输出拼接后降维)
        self.attn_fusion = nn.Linear(hidden_dim * 4, hidden_dim)

        # 3. 特征级门控网络 (用于 Gate Branch)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 4. 最终归一化与预测头
        self.mm_layernorm = nn.LayerNorm(hidden_dim)
        self.head_pred = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """网络参数的正态/Xavier初始化策略"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, prot_feat, prot_mask, mol_feat, mol_mask, ph_feat, use_ph, temp_feat, use_temp):
        batch_size = prot_feat.shape[0]

        prot_embed = self.prot_projection(prot_feat)
        prot_embed = prot_embed + self.prot_pos_embed[:, :prot_embed.shape[1], :]
        prot_embed = torch.cat((self.prot_cls_token.repeat(batch_size, 1, 1), prot_embed), dim=1)
        p_mask = torch.cat((torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device), (prot_mask == 0)), dim=1)

        mol_embed = self.mol_projection(mol_feat)
        mol_embed = torch.cat((self.mol_cls_token.repeat(batch_size, 1, 1), mol_embed), dim=1)
        m_mask = torch.cat((torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device), (mol_mask == 0)), dim=1)

        ph_embed = self.ph_projection(self.rbf_ph(ph_feat)).unsqueeze(1) if use_ph.sum() != 0 else torch.zeros(
            batch_size, 1, self.hidden_dim).to(self.device)
        temp_embed = self.temp_projection(self.rbf_temp(temp_feat)).unsqueeze(
            1) if use_temp.sum() != 0 else torch.zeros(batch_size, 1, self.hidden_dim).to(self.device)

        ph_mask = (use_ph == 0).bool()
        temp_mask = (use_temp == 0).bool()

        p_emb, m_emb = self.input_norm(prot_embed), self.input_norm(mol_embed)
        ph_emb, temp_emb = self.input_norm(ph_embed), self.input_norm(temp_embed)

        # 记录各模态物理存在状态，供 MAG 过滤使用
        valid_masks = torch.cat([
            torch.ones(batch_size, 1, device=self.device),
            torch.ones(batch_size, 1, device=self.device),
            use_ph.float(),
            use_temp.float()
        ], dim=1)

        final_alphas = None

        for layer in self.fusion_layers:
            scaled_feats, alphas = layer['mag']([p_emb, m_emb, ph_emb, temp_emb], valid_masks)
            sp_emb, sm_emb, sph_emb, stemp_emb = scaled_feats
            final_alphas = alphas

            src_prot = torch.cat([sm_emb, sph_emb, stemp_emb], dim=1)
            mask_prot = torch.cat([m_mask, ph_mask, temp_mask], dim=1)
            new_p = layer['prot'](p_emb, src_prot, key_padding_mask=mask_prot)

            src_mol = torch.cat([sp_emb, sph_emb, stemp_emb], dim=1)
            mask_mol = torch.cat([p_mask, ph_mask, temp_mask], dim=1)
            new_m = layer['mol'](m_emb, src_mol, key_padding_mask=mask_mol)

            src_ph = torch.cat([sp_emb, sm_emb, stemp_emb], dim=1)
            mask_ph = torch.cat([p_mask, m_mask, temp_mask], dim=1)
            new_ph = layer['ph'](ph_emb, src_ph, key_padding_mask=mask_ph)

            src_temp = torch.cat([sp_emb, sm_emb, sph_emb], dim=1)
            mask_temp = torch.cat([p_mask, m_mask, ph_mask], dim=1)
            new_temp = layer['temp'](temp_emb, src_temp, key_padding_mask=mask_temp)

            p_emb, m_emb, ph_emb, temp_emb = new_p, new_m, new_ph, new_temp

        p_emb, m_emb = self.dropout(p_emb), self.dropout(m_emb)

        prot_cls_feat = p_emb[:, 0].unsqueeze(1) * final_alphas[:, 0].view(-1, 1, 1)
        mol_cls_feat = m_emb[:, 0].unsqueeze(1) * final_alphas[:, 1].view(-1, 1, 1)
        ph_token = ph_emb * final_alphas[:, 2].view(-1, 1, 1)
        temp_token = temp_emb * final_alphas[:, 3].view(-1, 1, 1)

        mm_embed = torch.cat((prot_cls_feat, mol_cls_feat, ph_token, temp_token), dim=1)
        mm_embed = self.dropout(mm_embed)

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

        # =====================================================================
        #  [Branch 1] Attention Branch (自注意力机制，挖掘隐性跨模态之间的关联)
        # =====================================================================
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

        # =====================================================================
        #  [Branch 2] Gate Branch (特征级精细过滤，去噪解耦)
        # =====================================================================
        gate = torch.sigmoid(self.gate_net(mm_moe))
        gate = gate * (1 - mm_mask.unsqueeze(2).float())  # 彻底屏蔽缺失模态的激活

        gated_feat = (gate * mm_moe).sum(dim=1)
        gate_norm = gate.sum(dim=1) + 1e-8
        gated_feat = gated_feat / gate_norm

        # =====================================================================
        #  [Branch 3] Mean Branch (基础特征均衡混合)
        # =====================================================================
        valid_count = (1 - mm_mask.float()).sum(dim=1).unsqueeze(1) + 1e-8
        sum_feat = (mm_moe * (1 - mm_mask.unsqueeze(2).float())).sum(dim=1)
        mean_feat = sum_feat / valid_count

        # =====================================================================
        #  三路混合相加
        # =====================================================================
        weighted_mm = attn_feat + gated_feat + mean_feat
        final_mm_embed = self.dropout(self.mm_layernorm(weighted_mm))

        # 最终打分与去相关损失计算
        scores = self.head_pred(final_mm_embed)
        ortho_loss = calculate_ortho_loss(mm_embed)

        return scores, ortho_loss, moe_loss
# #2026831还是全局注意力这个模块
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# import random
# import numpy as np
# from model.module import MoE  # 确保你的环境里有这个模块
#
# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
#
#
# def calculate_ortho_loss(input_vec):
#     """
#     计算正交损失 (Orthogonal Loss)。
#     通过惩罚特征协方差矩阵的非对角线元素，促使不同模态提取到的特征相互独立（去相关），
#     从而减少冗余信息，提高多模态表征的丰富度。
#     """
#     x = input_vec - torch.mean(input_vec, axis=2).unsqueeze(2).repeat(1, 1, input_vec.shape[2])
#     cov_matrix = torch.matmul(x, x.transpose(1, 2)) / (x.shape[2] - 1)
#     loss = (torch.sum(cov_matrix ** 2) - torch.sum(torch.diagonal(cov_matrix, dim1=1, dim2=2) ** 2)) / (
#             cov_matrix.shape[0] * (cov_matrix.shape[1] - 1) * (cov_matrix.shape[2] - 1) + 1e-8)
#     return loss
#
#
# def temperature_scaled_softmax(logits, temperature=1.0, dim=0):
#     """带温度缩放的 Softmax，用于平滑或锐化最终模态融合的权重分布。"""
#     logits = logits / temperature
#     return torch.softmax(logits, dim=dim)
#
#
# class GaussianSmearing(nn.Module):
#     """
#     高斯涂抹 (Gaussian Smearing) 模块。
#     将一维的连续标量（如 pH 值、温度）映射到高维的高斯径向基函数 (RBF) 空间中，
#     增强模型对连续环境条件的表达能力。
#     """
#
#     def __init__(self, start=0.0, stop=1.0, num_gaussians=64):
#         super(GaussianSmearing, self).__init__()
#         offset = torch.linspace(start, stop, num_gaussians)
#         self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
#         self.register_buffer('offset', offset)
#
#     def forward(self, dist):
#         dist = dist.view(-1, 1) - self.offset.view(1, -1)
#         return torch.exp(self.coeff * torch.pow(dist, 2))
#
#
# class GlobalMAG(nn.Module):
#     """
#     全局模态自适应门控模块 (Modality-Adaptive Gating, MAG)。
#     基于论文公式 (1), (2), (3), (4) 实现，动态评估各模态数据质量并分配全局相对权重。
#     """
#
#     def __init__(self, hidden_dim, num_modalities=4, gate_th=0.1):
#         super(GlobalMAG, self).__init__()
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)
#         self.gate_th = gate_th
#
#     def forward(self, features_list, valid_masks):
#         g_scores = []
#         for x in features_list:
#             pooled = x.mean(dim=1)
#             h = F.relu(self.fc1(pooled))
#             g = torch.sigmoid(self.fc2(h))
#             g_scores.append(g)
#
#         g_tensor = torch.cat(g_scores, dim=1)
#         mask = (g_tensor >= self.gate_th) & valid_masks.bool()
#         g_masked = g_tensor.masked_fill(~mask, -1e9)
#         alphas = F.softmax(g_masked, dim=1)
#
#         out_features = []
#         for i, x in enumerate(features_list):
#             alpha_i = alphas[:, i].view(-1, 1, 1)
#             out_features.append(x * alpha_i)
#
#         return out_features, alphas
#
#
# class InterIntraFusionBlock(nn.Module):
#     """
#     模态内与模态间特征融合块。
#     """
#
#     def __init__(self, hidden_dim, num_heads=2, dropout=0.1):
#         super().__init__()
#         self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
#         self.norm1 = nn.LayerNorm(hidden_dim)
#
#         self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 4, hidden_dim)
#         )
#         self.norm3 = nn.LayerNorm(hidden_dim)
#
#     def forward(self, target_feat, source_feats, key_padding_mask=None):
#         cross_out, _ = self.cross_attn(query=target_feat, key=source_feats, value=source_feats,
#                                        key_padding_mask=key_padding_mask)
#         x = self.norm1(target_feat + cross_out)
#
#         self_out, _ = self.self_attn(query=x, key=x, value=x)
#         x = self.norm2(x + self_out)
#
#         ffn_out = self.ffn(x)
#         x = self.norm3(x + ffn_out)
#         return x
#
#
# class Enzyme(nn.Module):
#     def __init__(self, prot_dim=1024, mol_dim=300, rbf_dim=64, hidden_dim=128,
#                  batch_first=True, dropout=0.1, layers=4, expert_k=2, expert_total=10, device=torch.device('cuda')):
#         super(Enzyme, self).__init__()
#         self.device = device
#         self.hidden_dim = hidden_dim
#
#         # --- 模态特征初始化层 ---
#         self.prot_projection = nn.Linear(prot_dim, hidden_dim)
#         self.prot_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
#         self.prot_pos_embed = nn.Parameter(torch.zeros(1, 1024, hidden_dim))
#
#         self.mol_projection = nn.Linear(mol_dim, hidden_dim)
#         self.mol_cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
#
#         self.rbf_ph = GaussianSmearing(start=0.0, stop=14.0, num_gaussians=rbf_dim)
#         self.rbf_temp = GaussianSmearing(start=0.0, stop=100.0, num_gaussians=rbf_dim)
#         self.ph_projection = nn.Linear(rbf_dim, hidden_dim)
#         self.temp_projection = nn.Linear(rbf_dim, hidden_dim)
#
#         # --- 深度多层融合网络 ---
#         self.fusion_layers = nn.ModuleList([
#             nn.ModuleDict({
#                 'mag': GlobalMAG(hidden_dim, num_modalities=4),
#                 'prot': InterIntraFusionBlock(hidden_dim, num_heads=2, dropout=dropout),
#                 'mol': InterIntraFusionBlock(hidden_dim, num_heads=2, dropout=dropout),
#                 'ph': InterIntraFusionBlock(hidden_dim, num_heads=2, dropout=dropout),
#                 'temp': InterIntraFusionBlock(hidden_dim, num_heads=2, dropout=dropout)
#             }) for _ in range(layers)
#         ])
#
#         self.dropout = nn.Dropout(dropout)
#         self.input_norm = nn.LayerNorm(hidden_dim)
#
#         # --- 混合专家系统 (MoE) ---
#         self.moe = MoE(hidden_dim, hidden_dim, hidden_dim, expert_total, hidden_dim, noisy_gating=True, k=expert_k)
#
#
#         # 1. 全局检索探针 (用于 Attention Branch)
#         self.fusion_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
#
#         # 2. 全局信息池化注意力
#         self.final_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
#
#         # 3. 特征级门控网络 (用于 Gate Branch)
#         self.gate_net = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
#
#         # 4. 最终归一化与预测头
#         self.mm_layernorm = nn.LayerNorm(hidden_dim)
#         self.head_pred = nn.Linear(hidden_dim, 1)
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, module):
#         """网络参数的正态/Xavier初始化策略"""
#         if isinstance(module, nn.Linear):
#             nn.init.xavier_uniform_(module.weight)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.LayerNorm):
#             nn.init.ones_(module.weight)
#             nn.init.zeros_(module.bias)
#
#     def forward(self, prot_feat, prot_mask, mol_feat, mol_mask, ph_feat, use_ph, temp_feat, use_temp):
#         batch_size = prot_feat.shape[0]
#
#         prot_embed = self.prot_projection(prot_feat)
#         prot_embed = prot_embed + self.prot_pos_embed[:, :prot_embed.shape[1], :]
#         prot_embed = torch.cat((self.prot_cls_token.repeat(batch_size, 1, 1), prot_embed), dim=1)
#         p_mask = torch.cat((torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device), (prot_mask == 0)), dim=1)
#
#         mol_embed = self.mol_projection(mol_feat)
#         mol_embed = torch.cat((self.mol_cls_token.repeat(batch_size, 1, 1), mol_embed), dim=1)
#         m_mask = torch.cat((torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device), (mol_mask == 0)), dim=1)
#
#         ph_embed = self.ph_projection(self.rbf_ph(ph_feat)).unsqueeze(1) if use_ph.sum() != 0 else torch.zeros(
#             batch_size, 1, self.hidden_dim).to(self.device)
#         temp_embed = self.temp_projection(self.rbf_temp(temp_feat)).unsqueeze(
#             1) if use_temp.sum() != 0 else torch.zeros(batch_size, 1, self.hidden_dim).to(self.device)
#
#         ph_mask = (use_ph == 0).bool()
#         temp_mask = (use_temp == 0).bool()
#
#         p_emb, m_emb = self.input_norm(prot_embed), self.input_norm(mol_embed)
#         ph_emb, temp_emb = self.input_norm(ph_embed), self.input_norm(temp_embed)
#
#         # 记录各模态物理存在状态，供 MAG 过滤使用
#         valid_masks = torch.cat([
#             torch.ones(batch_size, 1, device=self.device),
#             torch.ones(batch_size, 1, device=self.device),
#             use_ph.float(),
#             use_temp.float()
#         ], dim=1)
#
#         final_alphas = None
#
#         for layer in self.fusion_layers:
#             scaled_feats, alphas = layer['mag']([p_emb, m_emb, ph_emb, temp_emb], valid_masks)
#             sp_emb, sm_emb, sph_emb, stemp_emb = scaled_feats
#             final_alphas = alphas
#
#             src_prot = torch.cat([sm_emb, sph_emb, stemp_emb], dim=1)
#             mask_prot = torch.cat([m_mask, ph_mask, temp_mask], dim=1)
#             new_p = layer['prot'](p_emb, src_prot, key_padding_mask=mask_prot)
#
#             src_mol = torch.cat([sp_emb, sph_emb, stemp_emb], dim=1)
#             mask_mol = torch.cat([p_mask, ph_mask, temp_mask], dim=1)
#             new_m = layer['mol'](m_emb, src_mol, key_padding_mask=mask_mol)
#
#             src_ph = torch.cat([sp_emb, sm_emb, stemp_emb], dim=1)
#             mask_ph = torch.cat([p_mask, m_mask, temp_mask], dim=1)
#             new_ph = layer['ph'](ph_emb, src_ph, key_padding_mask=mask_ph)
#
#             src_temp = torch.cat([sp_emb, sm_emb, sph_emb], dim=1)
#             mask_temp = torch.cat([p_mask, m_mask, ph_mask], dim=1)
#             new_temp = layer['temp'](temp_emb, src_temp, key_padding_mask=mask_temp)
#
#             p_emb, m_emb, ph_emb, temp_emb = new_p, new_m, new_ph, new_temp
#
#         p_emb, m_emb = self.dropout(p_emb), self.dropout(m_emb)
#
#         prot_cls_feat = p_emb[:, 0].unsqueeze(1) * final_alphas[:, 0].view(-1, 1, 1)
#         mol_cls_feat = m_emb[:, 0].unsqueeze(1) * final_alphas[:, 1].view(-1, 1, 1)
#         ph_token = ph_emb * final_alphas[:, 2].view(-1, 1, 1)
#         temp_token = temp_emb * final_alphas[:, 3].view(-1, 1, 1)
#
#         mm_embed = torch.cat((prot_cls_feat, mol_cls_feat, ph_token, temp_token), dim=1)
#         mm_embed = self.dropout(mm_embed)
#
#         mm_mask = torch.zeros(batch_size, 4).to(self.device)
#         mm_mask[:, 2] = (1 - use_ph).squeeze(1).float()
#         mm_mask[:, 3] = (1 - use_temp).squeeze(1).float()
#
#         mm_moe = torch.zeros(batch_size * 4, self.hidden_dim).to(self.device)
#         valid_indices = (mm_mask.reshape(-1) == 0)
#
#         if valid_indices.sum() > 0:
#             input_feat = mm_embed.reshape(-1, self.hidden_dim)[valid_indices]
#             tmp_moe, moe_loss = self.moe(input_feat, input_feat)
#             mm_moe[valid_indices] = tmp_moe
#         else:
#             moe_loss = torch.tensor(0.0).to(self.device)
#
#         mm_moe = mm_moe.reshape(batch_size, 4, self.hidden_dim)
#
#
#         #  Attention Branch (全局 Query 检索，捕捉跨模态模式) 动态选择重点，挖掘隐性跨模态之间的关联
#         query = self.fusion_query.repeat(batch_size, 1, 1)
#         attn_out, _ = self.final_attn(
#             query=query,
#             key=mm_moe,
#             value=mm_moe,
#             key_padding_mask=mm_mask.bool()  # 忽略缺失模态
#         )
#         attn_feat = attn_out.squeeze(1)
#
#         #  Gate Branch (特征级精细过滤，去噪解耦)，特征维度维度上筛选噪声
#         gate = torch.sigmoid(self.gate_net(mm_moe))
#         gate = gate * (1 - mm_mask.unsqueeze(2).float())  # 彻底屏蔽缺失模态的激活
#
#         gated_feat = (gate * mm_moe).sum(dim=1)
#         gate_norm = gate.sum(dim=1) + 1e-8
#         gated_feat = gated_feat / gate_norm
#
#         #  Mean Branch 均衡所有模态的信息，平等的看待每个存在的模态特征，将所有模态的基础信息融合，保留多模态的原始联合特征信息，均值分支提供一套公平均衡的融合结果 将四个特征向量加起来求平均值，得到一个融合的向量
#         valid_count = (1 - mm_mask.float()).sum(dim=1).unsqueeze(1) + 1e-8
#         sum_feat = (mm_moe * (1 - mm_mask.unsqueeze(2).float())).sum(dim=1)
#         mean_feat = sum_feat / valid_count
#
#         #  三路混合相加
#         weighted_mm = attn_feat + gated_feat + mean_feat
#         # mean+attention的双分支
#         # query = self.fusion_query.repeat(batch_size, 1, 1)
#         # attn_out, _ = self.final_attn(
#         #     query=query,
#         #     key=mm_moe,
#         #     value=mm_moe,
#         #     key_padding_mask=mm_mask.bool()  # 忽略缺失模态
#         # )
#         # attn_feat = attn_out.squeeze(1)
#         #
#         # # [Branch 2] Mean Branch (残差稳健底座，修复 Bug 后的版本)
#         # valid_count = (1 - mm_mask.float()).sum(dim=1).unsqueeze(1) + 1e-8
#         # sum_feat = (mm_moe * (1 - mm_mask.unsqueeze(2).float())).sum(dim=1)
#         # mean_feat = sum_feat / valid_count
#         #
#         # # [Fusion] 双路混合相加
#         # weighted_mm = attn_feat + mean_feat
#
#         # LayerNorm 归一化，平抑三路叠加的量级差异 (非常重要)
#         # # ---------------attention+gate双分支-----------------------
#         # # [Branch 1] Attention Branch (全局 Query 检索，捕捉跨模态模式)
#         # query = self.fusion_query.repeat(batch_size, 1, 1)
#         # attn_out, _ = self.final_attn(
#         #     query=query,
#         #     key=mm_moe,
#         #     value=mm_moe,
#         #     key_padding_mask=mm_mask.bool()  # 忽略缺失模态
#         # )
#         # attn_feat = attn_out.squeeze(1)
#         #
#         # # [Branch 2] Gate Branch (特征级精细过滤，去噪解耦)
#         # gate = torch.sigmoid(self.gate_net(mm_moe))
#         # gate = gate * (1 - mm_mask.unsqueeze(2).float())  # 彻底屏蔽缺失模态的激活
#         #
#         # gated_feat = (gate * mm_moe).sum(dim=1)
#         # gate_norm = gate.sum(dim=1) + 1e-8
#         # gated_feat = gated_feat / gate_norm
#         # weighted_mm = attn_feat + gated_feat
#         final_mm_embed = self.dropout(self.mm_layernorm(weighted_mm))
#
#         # 最终打分与去相关损失计算
#         scores = self.head_pred(final_mm_embed)
#         ortho_loss = calculate_ortho_loss(mm_embed)
#
#         return scores, ortho_loss, moe_loss