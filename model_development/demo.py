import pandas as pd
from data_process.clean_data import clean_column_names

# feature_data = pd.read_csv('../features.csv')
# feature_data.columns = [clean_column_names(col) for col in feature_data.columns]
# feature_data = feature_data.drop(['serial_number', 'collect_time', 'manufacturer', 'vendor'], axis=1)
#
#     # 负样本下采样
# sample_0 = feature_data[feature_data['failure_tag'] == 0].sample(frac=0.005, random_state=123)
# sample = sample_0._append(feature_data[feature_data['failure_tag'] == 1])
#
# X_train, y_train = sample.iloc[:, :-1], sample['failure_tag']
#
# if 'failure_tag' in X_train.columns:
#     X_train = X_train.drop(columns='failure_tag')
#
# if 'failure_tag' in X_train.columns:
#     print("Error: 'failure_tag' should not be in the features set.")
#
# else:
#     print("'failure_tag' is correctly not included in X_train.")


df1 = pd.read_csv('../data/test_features.csv')
df2 = pd.read_csv('../data/train_features.csv')

# 获取两个数据集的列名
columns_df1 = set(df1.columns)
columns_df2 = set(df2.columns)

# 找出只存在于 df1 的列
unique_to_df1 = columns_df1.difference(columns_df2)
# 找出只存在于 df2 的列
unique_to_df2 = columns_df2.difference(columns_df1)

# 打印结果
print("只在测试集中的列:", unique_to_df1)
print("只在训练集中的列:", unique_to_df2)
print('--------------')
print('测试集：',df1.shape)
print('训练集：',df2.shape)

# print(df1['failure_tag'])

print('-------------------')

# # data1 = pd.read_csv('../merged_dataset.csv')
# data1 = pd.read_csv('../data/test.csv')
# data2 = pd.read_csv('../data/memory_sample_kernel_log_k12_round1_a_test.csv')
#
# print(data1.info())
# print('------------------')
# print(data1.columns[3:27])
# print('------------------')
# print(data2.info())