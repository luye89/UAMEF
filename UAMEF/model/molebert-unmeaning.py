import os
import torch
from MoleBERT.loader import mol_to_graph_data_obj_simple
from MoleBERT.model import GNN
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import torch as th

DATASET_DIR = "E:/jiangjing/multi-task/data124"
# 建议修改文件名以区分以前的池化版本，例如 mol_embedding_sequence.pt
OUTPUT_FILE = os.path.join(DATASET_DIR, "mol_embedding_un.pt")
MODEL_PATH = "E:/jiangjing/multi-task/datasets/MoleBERT/Mole-BERT.pth"
SOURCE_FILES = ["kcat.csv", "km.csv", "ki.csv"]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在加载 MoleBERT 模型: {MODEL_PATH} ...")
    # MoleBERT 默认维度是 300
    model = GNN(num_layer=5, emb_dim=300).to(device)

    if os.path.exists(MODEL_PATH):
        model.from_pretrained(MODEL_PATH)
    else:
        print(f"错误: 找不到模型权重文件 {MODEL_PATH}")
        return

    model.eval()

    # 收集所有唯一的 SMILES
    all_smiles_set = set()
    for file_name in SOURCE_FILES:
        file_path = os.path.join(DATASET_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在，跳过: {file_path}")
            continue

        # 读取文件 (适配 csv 和 excel)
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # 自动查找 SMILES 列
        smiles_col = None
        for col in ['reactant_smiles', 'substrate_smiles', 'SMILES']:
            if col in df.columns:
                smiles_col = col
                break

        if smiles_col:
            current_smiles = df[smiles_col].dropna().unique()
            all_smiles_set.update(current_smiles)
        else:
            print(f"  - {file_name}: 未找到 SMILES 列，跳过。")

    unique_smiles_list = list(all_smiles_set)

    # 计算特征并存入字典
    embeddings_dict = {}
    error_count = 0

    print("开始提取原子级特征序列...")
    for smiles in tqdm(unique_smiles_list):
        # 处理空字符串或异常字符
        if not isinstance(smiles, str) or smiles.strip() == "" or "*" in smiles:
            # 存入形状为 (1, 300) 的全0向量，保持维度统一性
            embeddings_dict[smiles] = torch.zeros(1, 300).float()
            continue

        try:
            # RDKit 解析
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("RDKit 解析失败")

            # 转为图数据
            data = mol_to_graph_data_obj_simple(mol)
            data = data.to(device)

            with torch.no_grad():
                # 模型前向传播，得到原子级表示
                # node_repr 形状为 (num_atoms, 300)
                node_repr = model(data)

                # --- 核心修改点：不再使用 .mean(dim=0) ---
                # 直接转回 CPU 并保持 (N, 300) 的原始形状
                mol_vec = node_repr.cpu().float()

            # 存入字典: Key=SMILES, Value=Tensor(原子数, 300)
            embeddings_dict[smiles] = mol_vec

        except Exception:
            # 如果出错，存入 (1, 300) 的全0向量
            embeddings_dict[smiles] = torch.zeros(1, 300).float()
            error_count += 1

    print(f"特征提取完成，共计 {len(embeddings_dict)} 条，出错 {error_count} 条。")
    print(f"正在保存特征字典至: {OUTPUT_FILE}")
    torch.save(embeddings_dict, OUTPUT_FILE)


# 将以下代码追加到 molebert.py 的末尾

_mol_model_realtime = None


def get_moleBERT_embedding_realtime(smiles):
    """用于 UI 界面的单条实时底物特征提取（懒加载模式）"""
    global _mol_model_realtime
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if _mol_model_realtime is None:
        print("正在拉起 MoleBERT 模型进行实时提取...")
        _mol_model_realtime = GNN(num_layer=5, emb_dim=300).to(device)
        if os.path.exists(MODEL_PATH):
            _mol_model_realtime.from_pretrained(MODEL_PATH)
        else:
            print(f"警告：未找到 MoleBERT 权重 {MODEL_PATH}")
        _mol_model_realtime.eval()

    # 异常输入处理
    if not isinstance(smiles, str) or smiles.strip() == "" or "*" in smiles:
        return torch.zeros(1, 300).float()

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(1, 300).float()

        data = mol_to_graph_data_obj_simple(mol)
        data = data.to(device)

        with torch.no_grad():
            node_repr = _mol_model_realtime(data)
            mol_vec = node_repr.cpu().float()

        return mol_vec
    except Exception as e:
        print(f"MoleBERT 实时提取失败: {e}")
        return torch.zeros(1, 300).float()
if __name__ == "__main__":
    main()
