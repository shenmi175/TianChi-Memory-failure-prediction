import pandas as pd
from model_development.RF_LGB_regression import train_models

test_data = pd.read_csv('../data/train_features.csv')
X_test = test_data.drop(columns=['serial_number', 'collect_time',])  # 调整列名以匹配训练时使用的列

rf,lgb = train_models()

print('模型训练完成,进行预测')

# 确保使用相同的特征集
if 'failure_tag' in X_test.columns:
    X_test = X_test.drop(columns='failure_tag')  # 确保不包括目标列

# 确保预测数据集的列与训练数据集的列一致
expected_columns = rf.feature_names_in_  # 这是 RandomForest 训练时的列名
missing_columns = set(expected_columns) - set(X_test.columns)
if missing_columns:
    raise ValueError(f"以下特征在测试集中缺失：{missing_columns}")
additional_columns = set(X_test.columns) - set(expected_columns)
if additional_columns:
    print(f"警告：测试集中包含训练集未见过的额外特征：{additional_columns}")
    X_test = X_test.drop(columns=additional_columns)
    print('已经删除额外特征')

# 进行预测
rf_predictions = rf.predict(X_test)
lgb_predictions = lgb.predict(X_test)

# 融合预测结果（这里使用简单平均）
combined_predictions = (rf_predictions + lgb_predictions) / 2

# 转换预测结果为竞赛提交格式
def prepare_submission(data, predictions):
    # 确保 collect_time 是 datetime 类型
    if not pd.api.types.is_datetime64_any_dtype(data['collect_time']):
        data['collect_time'] = pd.to_datetime(data['collect_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # 添加预测结果
    data['predicted_failure_days'] = predictions
    # 将天数转换为分钟，并转换为整数
    data['predicted_interval_minutes'] = (data['predicted_failure_days'] * 24 * 60).astype(int)
    # 格式化 collect_time 为字符串
    data['formatted_collect_time'] = data['collect_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # 创建最终的 DataFrame
    submission = data[['serial_number', 'formatted_collect_time', 'predicted_interval_minutes']]

    return submission

# 创建一个包含需要的列的新 DataFrame
test_data = pd.DataFrame({
    'serial_number': test_data['serial_number'],
    'collect_time': test_data['collect_time']
})

submission_data = prepare_submission(test_data, combined_predictions)

# 保存到 CSV 文件，无头部，无索引
submission_data.to_csv('submission.csv', index=False, header=False)


