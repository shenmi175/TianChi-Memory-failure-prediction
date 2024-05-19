import pandas as pd

p = pd.read_csv('submission.csv')

print(p.shape)

print((p.iloc[:, 2] == 0).sum())