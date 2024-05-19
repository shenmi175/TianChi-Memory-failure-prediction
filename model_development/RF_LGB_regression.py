import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from data_process.clean_data import clean_column_names

def train_models():
    # 加载和准备特征数据
    feature_data = pd.read_csv('../data/train_features.csv')
    feature_data.columns = [clean_column_names(col) for col in feature_data.columns]
    feature_data = feature_data.drop(['serial_number', 'collect_time','manufacturer_x','vendor_x','tag','manufacturer_y','vendor_y',], axis=1)

    # 对负样本进行下采样以平衡数据集
    sample_0 = feature_data[feature_data['failure_tag'] == 0].sample(frac=0.005, random_state=123)
    sample = pd.concat([sample_0, feature_data[feature_data['failure_tag'] == 1]])

    X_train, y_train = sample.iloc[:, :-1], sample['failure_tag']

    if 'failure_tag' in X_train.columns:
        X_train = X_train.drop(columns='failure_tag')

    # 初始化 LightGBM 回归器
    lgb = LGBMRegressor(random_state=42,
                        n_estimators=228,
                        max_depth=-1)
    lgb.fit(X_train, y_train)

    # 初始化随机森林回归器
    rf = RandomForestRegressor(n_estimators=228,
                               random_state=42,
                               max_depth=None)
    rf.fit(X_train, y_train)

    return rf, lgb

if __name__ == '__main__':
    rf, lgb = train_models()
    print('模型训练完成')
