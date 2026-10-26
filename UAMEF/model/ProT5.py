# 由于unikp文件夹下的py文件存在相互引用，不将unikp文件夹加入sys.path将会出现模块未找到报错
# 正规操作中，应该将unikp转化为python模块，添加init.py文件声明。这里只是采取一个替代手段
import sys

import pandas as pd
from tqdm import tqdm
import gc
import re
import torch as th
from transformers import T5EncoderModel, T5Tokenizer
import os


def Seq_to_vec(Sequence):
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            #如果序列长度超过1000，采取掐头去尾：保留前500个和后500个
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
    sequences_Example = []
    #在每个氨基酸之间添加一个空格，prot5的Tokenizer需要将每个氨基酸视为一个单独的token
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)
    #加载T5的分词器，不进行小写转换
    tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
    #加载T5的编码器模型（encoder），只需要提取特征，不需要生成序列
    model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")
    gc.collect()
    #打印cuda是否可用（调试用）
    print(th.cuda.is_available())
    # 'cuda:0' if th.cuda.is_available() else
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    model = model.to(device)
    #将模型设置为评估模式，这会关闭dropout等训练才用的层，保证输出结果稳定
    model = model.eval()
    # features = []
    #定义字典用于存储结果
    features_dict = {}
    for i in tqdm(range(len(sequences_Example)), desc="Extracting"):
        # print('For sequence ', str(i+1)) # 用 tqdm 替代了刷屏的 print

        # 将非标准氨基酸（U,Z,O,B）替换为‘X’
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        # 将生成的词索引和掩码转为 Tensor 并移到 GPU
        input_ids = th.tensor(ids['input_ids']).to(device)
        attention_mask = th.tensor(ids['attention_mask']).to(device)
        # 禁用梯度计算
        with th.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        # 取出最后一层隐藏状态，维度通常是 [1, 序列长度, 1024]
        # 移回 CPU，但不转 Numpy
        embedding = embedding.last_hidden_state.cpu()
        # 因为 batch_size=1，直接取第0个
        seq_num = 0
        # 计算真实序列长度 (attention_mask中1的个数)
        seq_len = (attention_mask[seq_num] == 1).sum()

        # 取出有效部分的 embedding，并且去掉末尾的 EOS token (seq_len - 1)
        seq_emd = embedding[seq_num][:seq_len - 1]
        #存入字典
        raw_key = Sequence[i]
        features_dict[raw_key] = seq_emd.float()
    return features_dict
# 将以下代码追加到 protT5.py 的末尾

_prot_model_realtime = None
_prot_tokenizer_realtime = None


def get_protT5_embedding_realtime(sequence):
    """用于 UI 界面的单条实时特征提取（懒加载模式）"""
    global _prot_model_realtime, _prot_tokenizer_realtime
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

    # 懒加载：只有第一次调用时才加载庞大的 T5 模型
    if _prot_model_realtime is None:
        print("正在拉起 ProtT5 模型进行实时提取 (初次加载可能需要十几秒)...")
        _prot_tokenizer_realtime = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
        _prot_model_realtime = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50").to(device)
        _prot_model_realtime = _prot_model_realtime.eval()

    # 处理过长序列
    if len(sequence) > 1000:
        sequence = sequence[:500] + sequence[-500:]

    # 格式化序列
    seq_spaced = " ".join(list(sequence))
    seq_spaced = re.sub(r"[UZOB]", "X", seq_spaced)

    ids = _prot_tokenizer_realtime.batch_encode_plus([seq_spaced], add_special_tokens=True, padding=True)
    input_ids = th.tensor(ids['input_ids']).to(device)
    attention_mask = th.tensor(ids['attention_mask']).to(device)

    with th.no_grad():
        embedding = _prot_model_realtime(input_ids=input_ids, attention_mask=attention_mask)

    seq_len = (attention_mask[0] == 1).sum()
    seq_emd = embedding.last_hidden_state[0][:seq_len - 1].cpu()

    return seq_emd.float()

if __name__ == "__main__":
        DATA_DIR = "./mergedata"
        OUTPUT_PATH = os.path.join(DATA_DIR, "prot_embedding.pt")
        SOURCE_FILES = ["kcat.csv", "km.csv", "ki.csv"]
        all_sequences = set()

        for filename in SOURCE_FILES:
            file_path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(file_path):
                print(f"跳过不存在的文件: {filename}")
                continue

            # 读取文件
            if filename.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            # 提取序列列 (列名是 'sequence')
            if 'Sequence' in df.columns:
                seqs = df['Sequence'].dropna().unique()
                all_sequences.update(seqs)
            else:
                print(f"Warning: {filename} 中没有找到 'Sequence' 列")

        unique_seq_list = list(all_sequences)

        if len(unique_seq_list) > 0:
            # 调用函数
            prot_dict = Seq_to_vec(unique_seq_list)

            #  保存结果
            print(f"正在保存特征字典至: {OUTPUT_PATH}")
            th.save(prot_dict, OUTPUT_PATH)
        else:
            print("未找到任何序列，请检查文件路径。")

