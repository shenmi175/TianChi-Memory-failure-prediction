from model_development import RF_LGB
# from data_process import process_data_test
import pandas as pd
import os
from model_development.RF_LGB import train_models
from data_process.clean_data import clean_column_names

test = pd.read_csv('../data/test_features.csv')

test.columns = [clean_column_names(col) for col in test.columns]

group_min_test = test.drop(['serial_number', 'collect_time', 'manufacturer', 'vendor'], axis=1)

# 获取训练好的模型
rf, lgb = train_models()
print('模型训练完成,进行预测')

# 确保使用相同的特征集
if 'failure_tag' in test.columns:
    group_min_test = group_min_test.drop(columns='failure_tag')  # 确保不包括目标列

# 确保预测数据集的列与训练数据集的列一致
expected_columns = rf.feature_names_in_  # 这是 RandomForest 训练时的列名
missing_columns = set(expected_columns) - set(group_min_test.columns)
if missing_columns:
    raise ValueError(f"以下特征在测试集中缺失：{missing_columns}")
additional_columns = set(group_min_test.columns) - set(expected_columns)
if additional_columns:
    print(f"警告：测试集中包含训练集未见过的额外特征：{additional_columns}")
    group_min_test = group_min_test.drop(columns=additional_columns)



# 进行预测
rf_predictions = rf.predict_proba(group_min_test)[:, 1]
lgb_predictions = lgb.predict_proba(group_min_test)[:, 1]

# 进行简单平均融合
ensemble_probs = (rf_predictions + lgb_predictions) / 2

# 设置阈值
threshold = 0.5

# 根据平均融合的概率计算最终的预测结果
# 概率大于阈值的标记为类别 1，否则标记为类别 0
ensemble_predictions = (ensemble_probs > threshold).astype(int)


group_min_sn_test = pd.DataFrame(test[['serial_number', 'collect_time']])
# 将最终的预测结果添加到 group_min_sn_test DataFrame
group_min_sn_test['predict'] = ensemble_predictions


# 保存结果
group_min_sn_test = group_min_sn_test[group_min_sn_test['predict'] == 1]
group_min_sn_res = group_min_sn_test.drop('predict', axis=1)

output_path = os.path.join('', 'predict.csv')
print(group_min_sn_res)


# 由于预测pti对分数影响不大，先直接末尾增加pti为1
pti = 5
with open(output_path, 'w') as result_fp:
    for _, _row in group_min_sn_res.iterrows():
        result_fp.write("{},{},{}\n".format(_row.serial_number, _row.collect_time, pti))