import pandas as pd
from tsfresh.feature_extraction.settings import EfficientFCParameters
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute


def main():
    try:
        df = pd.read_csv('../merged_dataset.csv', low_memory=False)
        df['collect_time'] = pd.to_datetime(df['collect_time'])
        failure_tags = df['failure_tag']
        df = df.drop(['manufacturer', 'vendor', 'failure_time', 'failure_tag'], axis=1)

        # 使用 EfficientFCParameters 生成较少的特征
        settings = EfficientFCParameters()

        extracted_features = extract_features(df, column_id='serial_number', column_sort='collect_time',
                                              default_fc_parameters=settings, n_jobs=5, chunksize=64)

        imputed_features = impute(extracted_features)
        relevant_features = select_features(imputed_features, failure_tags)
        relevant_features.to_csv('extracted_features.csv', index=False)
        print(relevant_features.head())

    except Exception as e:
        print("Error occurred:", e)


if __name__ == '__main__':
    main()
