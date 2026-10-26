import numpy as np
import torch as th
# 模型构建需要的包
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F
# 借助scipy实现分段三次埃尔米特插值多项式
from scipy.interpolate import PchipInterpolator
# 借助statsmodels库来实现调整箱线图，需要安装：pip install statsmodels
from statsmodels.stats.stattools import medcouple

# DensityWeightMSELoss使用的样本权重函数作者已经开源并上传到了PyPI，因此可以用pip直接安装denseweight
from denseweight import DenseWeight
from scipy.stats import gaussian_kde





class ASERLoss(nn.Module):
    def __init__(self, all_targets_of_train_set: np.ndarray, k: float = 1.0):
        """
        ASER (Adapted Squared Error Relevance) 平衡改进版
        结合了 KDE 的频率感知能力与有界映射的稳定性。

        :param all_targets_of_train_set: 训练集所有目标值，用于拟合全局密度分布
        :param k: 差距控制系数。
                  k=1.0 时，最稀有样本的原始权重是普通样本的 2 倍；

        """
        super(ASERLoss, self).__init__()
        self.k = k

        # 1. 拟合高斯核密度估计 (KDE)
        # 过滤无效值确保拟合正常
        y_train = all_targets_of_train_set.flatten()
        y_train = y_train[np.isfinite(y_train)]
        self.kde = gaussian_kde(y_train)

        # 2. 预计算训练集的“稀有度基础” f(y) = 1/p(y)
        train_densities = self.kde(y_train)
        f_train = 1.0 / np.maximum(train_densities, 1e-10)

        # 3. 获取 f(y) 的极值，用于 Equation 6 的归一化映射
        # 这一步将无限的 1/p(y) 压缩到 [0, 1] 的相关性评分 phi 中
        self.f_min = np.min(f_train)
        self.f_max = np.max(f_train)

        # 4. 预计算训练集的平均权重，用于后续的均值归一化
        # 这一步是保证多任务学习中各任务 Loss 量级稳定的核心
        phi_train = (f_train - self.f_min) / (self.f_max - self.f_min + 1e-8)
        w_prime_train = 1 + self.k * phi_train
        self.avg_w_prime = np.mean(w_prime_train)

    def forward(self, predicts: th.Tensor, targets: th.Tensor) -> th.Tensor:
        # 确保输入展平
        predicts = predicts.flatten()
        targets = targets.flatten()
        targets_np = targets.detach().cpu().numpy()

        # 步骤 1: 计算当前 Batch 的密度及倒数 f(y)
        batch_densities = self.kde(targets_np)
        f_batch = 1.0 / np.maximum(batch_densities, 1e-10)

        # 步骤 2: 归一化映射得到 phi (相关性评分)
        # phi = 0 表示最常见（普通），phi = 1 表示最罕见（稀有）
        phi = (f_batch - self.f_min) / (self.f_max - self.f_min + 1e-8)
        phi = np.clip(phi, 0.0, 1.0)

        # 步骤 3: 计算原始权重 w' = 1 + k * phi
        # 普通样本权重保底为 1.0，稀有样本权重最高为 1 + k
        w_prime = 1 + self.k * phi

        # 步骤 4: 均值归一化
        # 确保该 Batch 的平均权重尺度与训练集基准对齐
        weights = w_prime / self.avg_w_prime

        # 转换为 Tensor 并执行加权 MSE 计算
        weights_th = th.from_numpy(weights).to(predicts.device, dtype=predicts.dtype)

        # 使用标准的平方误差内核
        squared_diff = th.pow(predicts - targets, 2)

        return th.mean(weights_th * squared_diff)