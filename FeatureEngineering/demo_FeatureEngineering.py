import pandas as pd
# from data_process import prodess_data_train

data = pd.read_csv('../data/merged_dataset.csv', low_memory=False)

data['collect_time'] = pd.to_datetime(data['collect_time'])

data = data.drop(['failure_time',],axis=1)

# 确保数据按照serial_number和collect_time排序
data.sort_values(['serial_number', 'collect_time'], inplace=True)


# 创建一个空字典来存储构建的特征
features_dict = {}

# 滑动窗口的大小（以小时为单位）
window_sizes = [1, 6, 24]  # 窗口大小：1小时，6小时，24小时

# 需要计算特征的列（从第2列到第25列）
feature_columns = data.columns[1:25]
print(feature_columns)
print(data.info())