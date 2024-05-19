import pandas as pd
import os
import numpy as np



kernel_log_data_path = 'memory_sample_kernel_log_round1_a_train.csv'# 内核日志路径
failure_tag_data_path = 'memory_sample_failure_tag_round1_a_train.csv'# 故障标签表路径
PARENT_FOLDER = '../data' # 数据的相对路径目录

print('---------------')
# 计算每个agg_time区间的和
def etl(path, agg_time):
    data = pd.read_csv(os.path.join(PARENT_FOLDER, path))
    # 检查缺失值
    if data.isnull().sum().sum() > 0:
        data = data.fillna(method='ffill')  # 填充缺失值
    # 降低时间精度 向上取整
    data['collect_time'] = pd.to_datetime(data['collect_time']).dt.ceil(agg_time)
    group_data = data.groupby(['serial_number', 'collect_time'], as_index=False).agg('sum')
    return group_data

# 设置聚合时间粒度
AGG_VALUE = 5
AGG_UNIT = 'min'
AGG_TIME = str(AGG_VALUE) + AGG_UNIT

# 示例仅使用了kernel数据
group_min = etl(kernel_log_data_path, AGG_TIME)
failure_tag = pd.read_csv(os.path.join(PARENT_FOLDER, failure_tag_data_path))
failure_tag['failure_time'] = pd.to_datetime(failure_tag['failure_time'])

# 为数据打标
merged_data = pd.merge(group_min, failure_tag[['serial_number', 'failure_time']], how='left', on=['serial_number'])
merged_data['failure_tag'] = (merged_data['failure_time'].notnull()) & ((merged_data['failure_time']
                                                                         - merged_data['collect_time']).dt.seconds <= AGG_VALUE * 60)
merged_data['failure_tag'] = merged_data['failure_tag'] + 0
# feature_data = merged_data.drop(['serial_number', 'collect_time', 'manufacturer', 'vendor', 'failure_time'], axis=1)

merged_data.to_csv('merged_dataset.csv')

print('已经合并数据')