import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from data_process.clean_data import clean_column_names

def train_models():
    feature_data = pd.read_csv('../data/train_features.csv')
    feature_data.columns = [clean_column_names(col) for col in feature_data.columns]
    feature_data = feature_data.drop(['serial_number', 'collect_time', 'manufacturer', 'vendor'], axis=1)

    # 负样本下采样
    sample_0 = feature_data[feature_data['failure_tag'] == 0].sample(frac=0.005, random_state=123)
    sample = sample_0._append(feature_data[feature_data['failure_tag'] == 1])

    X_train, y_train = sample.iloc[:, :-1], sample['failure_tag']

    if 'failure_tag' in X_train.columns:
        X_train = X_train.drop(columns='failure_tag')

    lgb = LGBMClassifier(random_state=42,
                         n_estimators=228,
                         max_depth=-1,
                         # num_leaves = 128,
                         # boosting_type = 'rf',
                         # bagging_freq=1,        # 每一次迭代都进行bagging
                         # bagging_fraction=0.8,  # 每次迭代使用80%的数据
                         # feature_fraction=0.8,   # 每次迭代使用80%的特征
                         # lambda_l2=0.1,
                         )
    lgb.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=128,  # 树的数量
        # criterion='gini',       # 用于分裂的质量度量，也可以是 'entropy'
        max_depth=None,  # 树的最大深度
        # min_samples_split=2,    # 分裂内部节点所需的最少样本数
        # min_samples_leaf=1,     # 在叶节点处需要的最少样本数
        # min_weight_fraction_leaf=0.0, # 在所有叶子节点处的权重总和中的最小加权分数
        # max_features='sqrt',    # 寻找最佳分割时考虑的特征数量
        # max_leaf_nodes=None,    # 以最佳优先方式增长树时的最大叶子节点数
        # min_impurity_decrease=0.0,    # 如果节点分裂导致不纯度减少大于或等于此值，则分裂节点
        # bootstrap=True,         # 是否在构建树时使用bootstrap样本
        oob_score=False,  # 是否使用袋外样本来估计泛化精度
        # n_jobs=-1,              # 拟合和预测时并行运行的作业数
        random_state=42,  # 控制组件的随机性
        verbose=0,  # 控制拟合和预测的冗长程度
        # warm_start=True,       # 设置为True时，重用上一个调用的解决方案以适应并在集合中添加更多的估计器
        # class_weight=weight_dict,      # 类别的权重
        # ccp_alpha=0.0,          # 用于最小成本-复杂性剪枝的复杂性参数
        # max_samples=None        # 如果 bootstrap 为 True，从 X 抽取的样本数
    )
    rf.fit(X_train, y_train)

    return rf, lgb

if __name__ == '__main__':
    rf, lgb = train_models()
    print('模型训练完成')

