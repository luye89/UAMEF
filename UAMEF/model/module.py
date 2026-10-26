import collections.abc
from itertools import repeat
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


def generate_cross_modal_mask(total_len, idx_prot, idx_mol, idx_ph, idx_temp, device):

    # 1. 初始化全为 -inf (严厉禁止所有交互)
    mask = torch.full((total_len, total_len), float("-inf"), device=device)
    mask.fill_diagonal_(0.0)
    mask[1:5, idx_prot:idx_ph] = 0.0
    mask[3, idx_ph] = 0.0
    # 假设 Fusion Token 3 (idx=4) 是专门负责 Temp 的
    mask[4, idx_temp] = 0.0
    # 允许蛋白质看蛋白质
    mask[idx_prot:idx_mol, idx_prot:idx_mol] = 0.0
    # 允许分子看分子
    mask[idx_mol:idx_ph, idx_mol:idx_ph] = 0.0
    # pH 看 pH (虽然对角线已设，但如果是序列则需要区域)
    mask[idx_ph, idx_ph] = 0.0
    # Temp 看 Temp
    mask[idx_temp, idx_temp] = 0.0

    return mask
class SparseDispatcher(object):
    def __init__(self, num_experts, gates):

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.relu(out)
        return out


class MoE(nn.Module):
    def __init__(self, query_size, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.query_size = query_size
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.task_gate = nn.Parameter(torch.zeros(query_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/(noise_stddev+1e-6))
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, task, x, train, noise_epsilon=1e-2):#noise_epsilon：噪声标准差最小值（防止标准差为 0，防止标准差为0
        modal_logits = x @ self.w_gate#输入特征与门控权重矩阵self.w_gate做矩阵乘法，计算每个样本对每个专家的初始特征
        task_logits = task @ self.task_gate#计算任务相关的门控得分，任务维度的专家得分
        clean_logits = modal_logits + task_logits#无噪声的专家得分
        if self.noisy_gating and train:#添加噪声，打散样本分配，避免专家负载不均
            raw_noise_stddev = x @ self.w_noise#计算噪声的原始强度
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))#确保噪声标准差为正，softplus将原始值转换为非负值，+noise_epsilon确保标准差严格>0
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)#注入高斯噪声，生成与clean_logits形状相同的标准正态分布噪声（torch.randn_like），乘以标准差后加到干净得分上，得到带噪声的专家得分。
            logits = noisy_logits#带噪声的得分作为最后的得分
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]#仅保留每个样本要分配的 Top-K 个专家的得分
        top_k_indices = top_indices[:, :self.k]#前k个索引，得到每个样本最终分配的 k 个专家的编号
        top_k_gates = self.softmax(top_k_logits)#对Top-K得分做 softmax，得到每个样本分配给这k个专家的权重，权重和为1
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)#计算每个样本的每个专家被选入 Top-K 的概率，sum(0) 按列求和,得到每个专家的负载，所有样本分配到该专家的概率和
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, query, x, loss_coef=1e-2):
        gates, load = self.noisy_top_k_gating(query, x, self.training)
        # calculate importance loss
        importance = gates.sum(0)#计算重要性，每个专家的权重之和

        loss = self.cv_squared(importance) + self.cv_squared(load)#变异系数的平方和
        loss *= loss_coef#缩放损失，负载均衡损失知识辅助损失，避免该过模型的主任务损失

        dispatcher = SparseDispatcher(self.num_experts, gates)#负责将输入分配给被选中的专家
        expert_inputs = dispatcher.dispatch(x)#列表，长度等于专家数，每个元素是对应专家要处理的输入数据，每个组合的特征送给选中的专家
        gates = dispatcher.expert_to_gates()#专家视角的权重
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]#每个元素是对应专家处理后的输出，专家值处理选中自己的组合
        y = dispatcher.combine(expert_outputs)#把各个专家的输出，按照门控权重加权整合，获得最后输入
        return y, loss


