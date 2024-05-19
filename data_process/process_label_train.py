import pandas as pd


kernel_log = pd.read_csv('../data/memory_sample_kernel_log_round1_a_train.csv')
failure_tag = pd.read_csv('../data/memory_sample_failure_tag_round1_a_train.csv')

kernel_log['collect_time'] = pd.to_datetime(kernel_log['collect_time'], format='%Y-%m-%d %H:%M:%S')
failure_tag['failure_time'] = pd.to_datetime(failure_tag['failure_time'], format='%Y/%m/%d %H:%M')

kernel_log_sorted = kernel_log.sort_values(['serial_number', 'collect_time'])
failure_tag_sorted = failure_tag.sort_values(['serial_number', 'failure_time'])


merged_data = pd.merge(kernel_log_sorted, failure_tag_sorted, on='serial_number')
merged_data = merged_data[merged_data['collect_time'] < merged_data['failure_time']]

merged_data['time_to_failure'] = (merged_data['failure_time'] - merged_data['collect_time']).dt.total_seconds() / 60

# 转换时间间隔从分钟到天
merged_data['gap_day'] = merged_data['time_to_failure'] / (60 * 24)


print('定义标签列')
# 定义一个函数来根据时间间隔打分
def assign_score(gap_day):
    if gap_day < 1:
        return 100
    elif 1 <= gap_day < 3:
        return 80
    elif 3 <= gap_day < 4:
        return 60
    elif 4 <= gap_day < 5:
        return 40
    elif 5 <= gap_day <= 7:
        return 20
    else:
        return 0

# 应用函数来创建新的列 'label'
merged_data['failure_tag'] = merged_data['gap_day'].apply(assign_score)

# 显示更新后的DataFrame
print(merged_data[['serial_number', 'collect_time', 'failure_time', 'time_to_failure', 'gap_day', 'failure_tag']].head())
merged_data = merged_data.drop(['time_to_failure', 'gap_day'], axis=1)

merged_data.fillna(0, inplace=True)

merged_data.to_csv('../data/merged_dataset.csv', index=False)