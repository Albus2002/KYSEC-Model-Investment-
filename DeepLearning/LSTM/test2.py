import h5py
import pandas as pd

# 定义文件路径和需要读取的行数
file_path = '/mnt/tmpdata/lvhanglong/data_20240312/df_facs_train_sample.h5'
num_rows = 5  # 你想要读取的行数
with h5py.File(file_path, 'r') as hdf:
    print(list(hdf['df'].keys()))
    dataset = hdf['df']['block2_items']
    data_partial = dataset[:5]  # 读取前五行数据

df_partial = pd.DataFrame(data_partial)
print(df_partial)
    
# 现在你可以使用变量 data，它包含了文件中前 num_rows 行的数据