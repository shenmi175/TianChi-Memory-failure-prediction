import pandas as pd


def convert_to_submission_format(predictions):


    # 将天数转换为分钟
    predictions['predicted_interval_minutes'] = predictions['predicted_interval_days'] * 24 * 60

    # 应用打分规则
    def assign_score(days):
        if days < 1:
            return 100
        elif 1 <= days < 3:
            return 80
        elif 3 <= days < 4:
            return 60
        elif 4 <= days < 5:
            return 40
        elif 5 <= days <= 7:
            return 20
        else:
            return 0

    predictions['score'] = predictions['predicted_interval_days'].apply(assign_score)

    # 格式化输出
    # 假设 `predicted_failure_date` 已经是正确的日期时间格式
    predictions['submission_format'] = predictions.apply(
        lambda
            x: f"{x['serial_number']},{x['predicted_failure_date']:%Y-%m-%d %H:%M:%S},{x['predicted_interval_minutes']}",
        axis=1
    )

    # 返回只包含所需提交列的DataFrame
    return predictions[['submission_format']]



data = pd.read_csv('../data/memory_sample_kernel_log_k12_round1_a_test.csv')

predictions_df = pd.DataFrame(data)

# 转换预测结果
submission_data = convert_to_submission_format(predictions_df)
print(submission_data)
