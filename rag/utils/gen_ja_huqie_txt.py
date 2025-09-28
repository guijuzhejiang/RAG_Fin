import pandas as pd

# 定义词性映射表
pos_mapping = {
    "名詞": "n",
    "動詞": "v",
    "形容詞": "a",
    "副詞": "d",
    "助詞": "p",
    "助動詞": "u",
    "接続詞": "c",
    "連体詞": "b",
    "感動詞": "e",
    "記号": "x",
    "接頭詞": "h",
    "フィラー": "f",
    "その他": "z",
    # 未知的原始词性可以映射到通用符号
    "default": "unk"  # 未知词性的默认映射
}

# 加载文件到 DataFrame
def load_file_to_dataframe(file_path):
    df = pd.read_csv(file_path, sep=" ", header=None, names=["word", "frequency", "pos"])
    return df

# 转换词性
def map_pos(df, column_name, mapping):
    df["mapped_pos"] = df[column_name].apply(lambda x: mapping.get(x, mapping["default"]))
    return df

# 文件路径
input_file_path = "/home/zzg/workspace/pycharm/RAGFlow/rag/res/ja_dict.txt"
output_file_path = "/home/zzg/workspace/pycharm/RAGFlow/rag/res/ja_huqie.txt"

# 加载原文件
df = load_file_to_dataframe(input_file_path)

# 执行词性映射
df = map_pos(df, "pos", pos_mapping)

# 保存到目标文件
df[["word", "frequency", "mapped_pos"]].to_csv(output_file_path, sep=" ", index=False, header=False)

# 打印转换后的词性集合
print("目标文件中的词性：", set(df["mapped_pos"]))
