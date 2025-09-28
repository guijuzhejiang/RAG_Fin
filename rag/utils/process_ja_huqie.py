import pandas as pd

# 读取文件并转换为 DataFrame
def load_file_to_dataframe(file_path):
    # 假设文件是空格分隔的格式
    df = pd.read_csv(file_path, sep=" ", header=None, names=["word", "frequency", "pos"])
    return df

# 获取第三列的唯一值
def get_unique_values(df, column_name):
    return set(df[column_name])

# 文件路径
# file_path = "/home/zzg/workspace/pycharm/RAGFlow/rag/res/ja_dict.txt"
file_path = "/rag/res/huqie.txt"

# 加载文件
df = load_file_to_dataframe(file_path)

# 查看第三列（pos）有哪些不同值
unique_pos = get_unique_values(df, "pos")

# 输出结果
print("第三列的不同值：", unique_pos)
