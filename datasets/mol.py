import os
import torch
import h5py
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from MoleBERT.loader import mol_to_graph_data_obj_simple
from MoleBERT.model import GNN

# --- 配置路径 ---
DATASET_DIR = r"E:\jiangjing\enzyme\data124"
TASKS = ["kcat", "km", "ki"]

# 基础底物特征的 H5 输出路径
BASE_H5_PATH = r"D:\data\mol_embedding.h5"

# MoleBERT 模型权重路径
MODEL_PATH = r"E:\jiangjing\multi-task\datasets\MoleBERT\Mole-BERT.pth"


def escape_smiles_for_h5(smiles):
    """
    HDF5 格式会将 '/' 视作文件夹（Group）的分隔符。
    SMILES 经常包含 '/' 和 '\'（用于表示双键的顺反异构等）。
    必须将它们替换掉，才能作为 H5 文件的正确键名。
    """
    return str(smiles).replace('/', '_FSLASH_').replace('\\', '_BSLASH_')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前计算设备: {device}")
    print(f"正在加载 MoleBERT 模型: {MODEL_PATH} ...")

    # MoleBERT 默认维度是 300
    model = GNN(num_layer=5, emb_dim=300).to(device)

    if os.path.exists(MODEL_PATH):
        model.from_pretrained(MODEL_PATH)
    else:
        print(f"错误: 找不到模型权重文件 {MODEL_PATH}")
        return

    model.eval()

    # 1. 扫描所有 CSV 文件，收集底物 SMILES
    all_smiles = set()
    for task in TASKS:
        task_dir = os.path.join(DATASET_DIR, task)
        if os.path.exists(task_dir):
            for file_name in os.listdir(task_dir):
                if file_name.endswith(".csv"):
                    file_path = os.path.join(task_dir, file_name)
                    print(f"正在扫描文件: {file_path}")
                    try:
                        df = pd.read_csv(file_path)
                        # 查找 SMILES 列
                        if 'SMILES' in df.columns:
                            current_smiles = df['SMILES'].dropna().unique()
                            all_smiles.update(current_smiles)
                    except Exception as e:
                        print(f"读取文件 {file_path} 失败: {e}")
        else:
            print(f"未找到路径: {task_dir}")

    print(f"\n共从 CSV 中发现 {len(all_smiles)} 种唯一底物 SMILES。")

    if len(all_smiles) == 0:
        print("未找到任何底物 SMILES，程序退出。")
        return

    # 2. 在线对比与特征提取
    error_count = 0
    new_extract_count = 0

    print(f"\n正在打开或创建特征库: {BASE_H5_PATH}")
    # 以追加模式打开 H5 文件，支持实时写入
    with h5py.File(BASE_H5_PATH, 'a') as h5_file:

        for smiles in tqdm(all_smiles, desc="Extracting & Saving SMILES"):
            # 将原始 SMILES 转义，作为 H5 的 Key
            h5_key = escape_smiles_for_h5(smiles)

            # 如果特征已经存在于 H5 中，直接跳过去重
            if h5_key in h5_file:
                continue

            new_extract_count += 1

            # 异常处理：空字符串等
            if not isinstance(smiles, str) or smiles.strip() == "" or "*" in smiles:
                mol_vec = np.zeros((1, 300), dtype=np.float32)
            else:
                try:
                    # RDKit 解析
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        raise ValueError("RDKit 解析失败")

                    # 转为图数据
                    data = mol_to_graph_data_obj_simple(mol)
                    data = data.to(device)

                    with torch.no_grad():
                        # 模型前向传播，得到原子级表示 (num_atoms, 300)
                        node_repr = model(data)
                        # 转为 numpy 数组 (HDF5 不支持直接存 PyTorch Tensor)
                        mol_vec = node_repr.cpu().numpy().astype(np.float32)

                except Exception:
                    # 如果出错，存入 (1, 300) 的全0向量
                    mol_vec = np.zeros((1, 300), dtype=np.float32)
                    error_count += 1

            # 3. 存入 H5 文件
            h5_file.create_dataset(
                name=h5_key,
                data=mol_vec,
                compression="gzip",
                compression_opts=4
            )

    print(f"\n提取完成！")
    print(f"本次新提取了 {new_extract_count} 条特征（其中出错 {error_count} 条）。")
    print(f"已安全保存至: {BASE_H5_PATH}")


if __name__ == "__main__":
    main()