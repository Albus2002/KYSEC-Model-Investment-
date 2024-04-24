import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# 定义文件夹路径和输出文件名
folder_path = '/mnt/tmpdata/lvhanglong/data_20240308'
output_csv = '1_3_sz_data.csv'

# 获取文件夹中所有以'300'或'301'开头并以'sz.h5'结尾的文件路径
h5_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('sz.h5') and (f.startswith('300') or f.startswith('301'))]

# 定义处理每个.h5文件的函数
def process_file(file_path):
    df = pd.read_hdf(file_path)
    # 抽样，抽样比例为 1/3
    df_sampled = df.sample(frac=1/3)
    df_sampled.fillna(method='bfill', inplace=True)
    df_sampled['Ticker'] = df_sampled['Ticker'].str.replace('.sz', '')
    df_sampled['Ticker'] = df_sampled['Ticker'].apply(lambda x: int(x))
    return df_sampled

# 使用 ProcessPoolExecutor 加速文件处理
with ProcessPoolExecutor(max_workers=72) as executor:
    # 提交所有文件处理任务
    future_to_file = {executor.submit(process_file, file_path): file_path for file_path in h5_files}
    results = []
    
    # 打印进度和处理结果
    for future in as_completed(future_to_file):
        file_path = future_to_file[future]
        try:
            result = future.result()
            results.append(result)
            print(f"Processed file: {os.path.basename(file_path)}")
        except Exception as exc:
            print(f"File {os.path.basename(file_path)} generated an exception: {exc}")

# 合并所有 DataFrame
combined_df = pd.concat(results, ignore_index=True)

# 将合并后的数据保存为CSV文件
combined_df.to_csv(output_csv, index=False)

# 打印完成消息
print(f"Data combined and saved to {output_csv}")