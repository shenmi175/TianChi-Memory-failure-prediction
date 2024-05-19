import pandas as pd

data = pd.read_csv('../data/test.csv', low_memory=False)

data['collect_time'] = pd.to_datetime(data['collect_time'])

# 确保数据按照serial_number和collect_time排序
data.sort_values(['serial_number', 'collect_time'], inplace=True)


# 创建一个空字典来存储构建的特征
features_dict = {}

# 滑动窗口的大小（以小时为单位）
window_sizes = [1, 6, 24]  # 窗口大小：1小时，6小时，24小时

# 需要计算特征的列（从第2列到第25列）
feature_columns = data.columns[3:24]


print('正在计算滑动窗口统计')
# 计算每个窗口大小的滑动窗口统计
for window in window_sizes:
    window_str = f"{window}H"
    for column in feature_columns:
        df_rolled = data.groupby('serial_number').rolling(window=window_str, on='collect_time')
        features_dict[f'sum_{column}_{window}h'] = df_rolled[column].sum().reset_index(level=0, drop=True).tolist()
        features_dict[f'mean_{column}_{window}h'] = df_rolled[column].mean().reset_index(level=0, drop=True).tolist()
        features_dict[f'std_{column}_{window}h'] = df_rolled[column].std().reset_index(level=0, drop=True).tolist()
        features_dict[f'min_{column}_{window}h'] = df_rolled[column].min().reset_index(level=0, drop=True).tolist()
        features_dict[f'max_{column}_{window}h'] = df_rolled[column].max().reset_index(level=0, drop=True).tolist()
        print('已计算特征',column)
    print(r'已经计算滑动窗口大小：',window)
print(r'滑动窗口大小计算完成')


print('正在保存数据')
features_df = pd.DataFrame(features_dict)
data.reset_index(drop=True, inplace=True)
features_df.reset_index(drop=True, inplace=True)
merged_data = pd.concat([data, features_df], axis=1)

# 保存数据
merged_data.reset_index(drop=True).to_csv('../data/test_features.csv', index=False)
features_dict = {}
print('数据保存完成')



print('正在提取时间点特征')
# 提取时间点特征
data['minute'] = data['collect_time'].dt.minute
data['hour'] = data['collect_time'].dt.hour
data['day'] = data['collect_time'].dt.day

# 将提取的特征添加到特征字典中
features_dict['minute'] = data['minute']
features_dict['hour'] = data['hour']
features_dict['day'] = data['day']
print('时间点特征提取完成')


print('正在保存数据')

df = pd.read_csv('../data/test_features.csv')
features_df = pd.DataFrame(features_dict)
df.reset_index(drop=True, inplace=True)
features_df.reset_index(drop=True, inplace=True)
merged_data = pd.concat([df, features_df], axis=1)

# 保存数据
merged_data.reset_index(drop=True).to_csv('../data/test_features.csv', index=False)
features_dict = {}
print('数据保存完成')


print('正在计算滞后时间窗口')
# 滞后时间窗口
lags = [1, 6, 12, 24]

# 遍历每个滞后窗口
for lag in lags:
    window_str = f"{lag}H"
    # 创建滚动窗口对象
    df_rolled = data.groupby('serial_number').rolling(window=window_str, on='collect_time')

    # 遍历需要计算的特征列
    for column in feature_columns:
        # 对每个特征列进行滚动窗口内的求和操作，并保存到字典中
        feature_name = f'lag_sum_{column}_{lag}h'  # 创建特征名
        features_dict[feature_name] = df_rolled[column].sum().reset_index(level=0, drop=True).tolist()
print('滞后时间窗口计算完成')



print('正在计算分钟滞后时间窗口')
# 滞后时间窗口列表，以分钟为单位
minute_lags = [1, 5, 15, 30]

# 遍历每个滞后窗口
for lag in minute_lags:
    window_str = f"{lag}min"
    # 创建滚动窗口对象
    df_rolled = data.groupby('serial_number').rolling(window=window_str, on='collect_time')

    # 遍历需要计算的特征列
    for column in feature_columns:
        # 对每个特征列进行滚动窗口内的求和操作，并保存到字典中
        feature_name = f'lag_sum_{column}_{lag}min'  # 创建特征名
        features_dict[feature_name] = df_rolled[column].sum().reset_index(level=0, drop=True).tolist()
        print('已经计算的滞后特征：',column)
print('分钟滞后时间窗口计算完成')


print('正在保存数据')

df = pd.read_csv('../data/test_features.csv')
features_df = pd.DataFrame(features_dict)
df.reset_index(drop=True, inplace=True)
features_df.reset_index(drop=True, inplace=True)
merged_data = pd.concat([df, features_df], axis=1)

# 保存数据
merged_data.reset_index(drop=True).to_csv('../data/test_features.csv', index=False)
features_dict = {}
print('数据保存完成')






print('正在计算一阶差分')

# 对feature_columns中的每个特征进行一阶差分
for column in feature_columns:
    features_dict[f'diff_{column}'] = data[column].diff().fillna(0)

print('一阶差分计算完成')


# print('正在计算时间变化率')
# # 定义变化率的时间窗口（以分钟为单位）
# change_periods = [5,15,30,60,120,240,360]  # 30分钟、60分钟、2小时
#
# # 遍历每个时间窗口和每个特征列来计算变化率
# for period in change_periods:
#     window_str = f"{period}min"
#     df_rolled = data.groupby('serial_number').rolling(window=window_str, on='collect_time')
#
#     for column in feature_columns:
#         # 计算窗口结束时点和开始时点的值差，除以窗口开始时点的值，得到变化率
#         rate_change = (df_rolled[column].apply(lambda x: x.iloc[-1] - x.iloc[0]) / df_rolled[column].apply(lambda x: x.iloc[0] if x.iloc[0] != 0 else 1)).reset_index(level=0, drop=True)
#         features_dict[f'rate_change_{column}_{window_str}'] = rate_change
#         print('已经计算的特征：',column)
#
#     print('已经计算的时间窗口:',period)
# print('时间变化率计算完成')
#
# print('正在保存数据')
#
# df = pd.read_csv('../data/test_features.csv')
# features_df = pd.DataFrame(features_dict)
# df.reset_index(drop=True, inplace=True)
# features_df.reset_index(drop=True, inplace=True)
# merged_data = pd.concat([df, features_df], axis=1)
#
# # 保存数据
# merged_data.reset_index(drop=True).to_csv('../data/test_features.csv', index=False)
# features_dict = {}
# print('数据保存完成')




# print('正在计算滚动窗口')
# # 定义多个滚动窗口大小
# rolling_periods = ['5min', '10min', '20min', '1D', '3D', '7D']  # 分钟和天
#
#
# # 遍历每个时间窗口和每个特征列来计算滚动窗口统计
# for period in rolling_periods:
#     df_rolled = data.groupby('serial_number').rolling(window=period, on='collect_time')
#
#     for column in feature_columns:
#         features_dict[f'rolling_mean_{column}_{period}'] = df_rolled[column].mean().reset_index(level=0, drop=True).tolist()
#         features_dict[f'rolling_std_{column}_{period}'] = df_rolled[column].std().reset_index(level=0, drop=True).tolist()
#         features_dict[f'rolling_min_{column}_{period}'] = df_rolled[column].min().reset_index(level=0, drop=True).tolist()
#         features_dict[f'rolling_max_{column}_{period}'] = df_rolled[column].max().reset_index(level=0, drop=True).tolist()
#         print('已经计算的特征:',column)
#
#     print('已经计算的滚动窗口大小:',period)
# print('滚动窗口计算完成')
#
# print('正在保存数据')
#
# df = pd.read_csv('../data/test_features.csv')
# features_df = pd.DataFrame(features_dict)
# df.reset_index(drop=True, inplace=True)
# features_df.reset_index(drop=True, inplace=True)
# merged_data = pd.concat([df, features_df], axis=1)
#
# # 保存数据
# merged_data.reset_index(drop=True).to_csv('../data/test_features.csv', index=False)
# features_dict = {}
# print('数据保存完成')

print('正在计算扩展窗口')
# 遍历每个特征列来计算扩展窗口统计
for column in feature_columns:
    df_expanded = data.groupby('serial_number')[column].expanding().agg(['mean', 'std', 'min', 'max']).reset_index(level=0, drop=True)
    features_dict[f'expanding_mean_{column}'] = df_expanded['mean']
    features_dict[f'expanding_std_{column}'] = df_expanded['std']
    features_dict[f'expanding_min_{column}'] = df_expanded['min']
    features_dict[f'expanding_max_{column}'] = df_expanded['max']
    print('已经计算的扩展窗口特征:',column)

print('扩展窗口特征计算完成')

print('正在保存数据')

df = pd.read_csv('../data/test_features.csv')
features_df = pd.DataFrame(features_dict)
df.reset_index(drop=True, inplace=True)
features_df.reset_index(drop=True, inplace=True)
merged_data = pd.concat([df, features_df], axis=1)

# 保存数据
merged_data.reset_index(drop=True).to_csv('../data/test_features.csv', index=False)
features_dict = {}
print('数据保存完成')



print('计算时间衰减特征')
# 对某个特征列应用指数加权移动平均
alpha = 0.3  # 衰减因子
for column in feature_columns:
    features_dict[f'ewm_{column}'] = data.groupby('serial_number')[column].transform(lambda x: x.ewm(alpha=alpha).mean())
    print('已计算特征：',column)
print('时间衰减特征计算完成')



print('正在保存数据')
print(len(features_dict.keys()))
print(features_dict.keys())

df = pd.read_csv('../data/test_features.csv')
features_df = pd.DataFrame(features_dict)
df.reset_index(drop=True, inplace=True)
features_df.reset_index(drop=True, inplace=True)
merged_data = pd.concat([df, features_df], axis=1)

# 保存数据
merged_data.reset_index(drop=True).to_csv('../data/test_features.csv', index=False)
features_dict = {}
print('数据保存完成')