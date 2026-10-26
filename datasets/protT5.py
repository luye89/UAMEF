import os
import re
import gc
import numpy as np
import pandas as pd
import torch as th
import h5py
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer


def extract_and_save_base_to_h5(sequences_to_extract, output_h5_path):
    """
    接收需要提取的基础序列列表，调用 T5 模型提取特征，并实时写入 H5 文件。
    """
    if len(sequences_to_extract) == 0:
        print("没有需要提取的新基础序列。")
        return

    print("正在加载 T5 模型和分词器...")
    tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")
    gc.collect()

    # 自动检测 CUDA 环境
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    print(f"当前使用的计算设备: {device}")
    model = model.to(device).eval()

    # 以追加模式 ('a') 打开或创建 H5 文件
    with h5py.File(output_h5_path, 'a') as h5_file:
        for original_seq in tqdm(sequences_to_extract, desc="Extracting & Saving Base Proteins"):
            # 防御性编程：如果该序列已经在当前要写入的 H5 中，则跳过
            if original_seq in h5_file:
                continue

            # 1. 长度截断逻辑：超过 1000 则掐头去尾
            processing_seq = original_seq
            if len(processing_seq) > 1000:
                processing_seq = processing_seq[:500] + processing_seq[-500:]

            # 2. 序列清洗：替换非标准氨基酸，添加空格隔开
            clean_seq = re.sub(r"[UZOB]", "X", processing_seq)
            spaced_seq = ' '.join(list(clean_seq))

            # 3. T5 编码与推理
            ids = tokenizer.batch_encode_plus([spaced_seq], add_special_tokens=True, padding=True)
            input_ids = th.tensor(ids['input_ids']).to(device)
            attention_mask = th.tensor(ids['attention_mask']).to(device)

            with th.no_grad():
                embedding = model(input_ids=input_ids, attention_mask=attention_mask)

            # 4. 提取有效长度特征并转为 Numpy 数组
            seq_len = (attention_mask[0] == 1).sum().item()
            seq_emd = embedding.last_hidden_state[0][:seq_len - 1].cpu().numpy().astype(np.float32)

            # 5. 存入 H5 文件
            h5_file.create_dataset(
                name=original_seq,
                data=seq_emd,
                compression="gzip",
                compression_opts=4
            )

    print(f"\n基础特征提取完成！已安全保存至: {output_h5_path}")


if __name__ == "__main__":
    # --- 配置路径 ---
    DATA_DIR = r"E:\jiangjing\enzyme\data124"
    TASKS = ["kcat", "km", "ki"]
    BASE_H5_PATH = r"D:\data\prot_embedding.h5"

    # 1. 扫描所有基础 CSV 文件（自动排除增强文件），收集所有基础序列
    all_base_sequences = set()
    for task in TASKS:
        task_dir = os.path.join(DATA_DIR, task)
        if os.path.exists(task_dir):
            for file_name in os.listdir(task_dir):
                # 仅读取基础数据集文件（排除包含 aug 或 augmentation 的增强文件）
                if file_name.endswith(".csv") and not ("aug" in file_name.lower() or "augmentation" in file_name.lower()):
                    file_path = os.path.join(task_dir, file_name)
                    print(f"正在扫描基础文件: {file_path}")
                    try:
                        df = pd.read_csv(file_path)
                        if 'EnzymeSequence' in df.columns:
                            seqs = df['EnzymeSequence'].dropna().unique()
                            all_base_sequences.update(seqs)
                    except Exception as e:
                        print(f"读取文件 {file_path} 失败: {e}")
        else:
            print(f"未找到路径: {task_dir}")

    print(f"\n共从基础数据集 CSV 中发现 {len(all_base_sequences)} 条唯一基础序列。")

    # 2. 与现有的 H5 特征库进行去重比对
    new_sequences_to_extract = []
    if os.path.exists(BASE_H5_PATH):
        print(f"发现现有特征库 {BASE_H5_PATH}，正在进行去重比对...")
        with h5py.File(BASE_H5_PATH, 'r') as base_h5:
            for seq in all_base_sequences:
                if seq not in base_h5:
                    new_sequences_to_extract.append(seq)
    else:
        print(f"未找到特征库 {BASE_H5_PATH}，将新建并提取所有基础序列。")
        new_sequences_to_extract = list(all_base_sequences)

    print(f"比对完毕：其中需要新提取特征的基础序列有 {len(new_sequences_to_extract)} 条。")

    # 3. 执行提取与保存
    extract_and_save_base_to_h5(new_sequences_to_extract, BASE_H5_PATH)