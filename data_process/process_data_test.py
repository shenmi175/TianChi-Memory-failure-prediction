import pandas as pd
import os


PARENT_FOLDER = '../data' # 数据的相对路径目录

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


# 测试数据
group_data_test = etl('memory_sample_kernel_log_k12_round1_a_test.csv', AGG_TIME)
# group_min_sn_test = pd.DataFrame(group_data_test[['serial_number', 'collect_time']])
# group_min_test = group_data_test.drop(['serial_number', 'collect_time', 'manufacturer', 'vendor'], axis=1)

group_data_test.to_csv('../data/test.csv')