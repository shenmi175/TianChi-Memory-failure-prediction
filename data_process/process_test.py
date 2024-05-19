import pandas as pd

data = pd.read_csv('../data/memory_sample_kernel_log_k12_round1_a_test.csv')

data.fillna(0,inplace=True)
data.to_csv('../data/test.csv')