# TianChi-Memory-failure-prediction
该项目来自[阿里天池的内存故障预测竞赛](https://tianchi.aliyun.com/competition/entrance/532055/information)

通过将前向填充与数据聚合相结合的方法改变数据分布，实现了很大的突破

最终代码是在[baseline](https://tianchi.aliyun.com/notebook/303693?spm=a2c22.12281978.0.0.30d447c99kJa2q)的基础上实现的

截至2024-4-16，分数为25.4148，排名7/3603

# 改进方法
- 先进行前向填充，再进行数据聚合
  
  **注意实现方式：**
```python
  # 计算每个agg_time区间的和
def etl(path, agg_time):
    data = pd.read_csv(os.path.join(PARENT_FOLDER, path))
    data = data.fillna(method='ffill')  # 填充缺失值
    # 降低时间精度 向上取整
    data['collect_time'] = pd.to_datetime(data['collect_time']).dt.ceil(agg_time)
    group_data = data.groupby(['serial_number', 'collect_time'], as_index=False).agg('sum')
    return group_data
```
- 提高欠采样多数类比例
- 将pti由1调整到了5
- 以上三种改进方式需要组合使用

## 效果/分数

由于在采样时没有控制其随机性，故分数是在一个范围内波动的。组合方案目前最高分数为29.37

|组合方式|效果/分数|
|:--:|:--:|
|原先的方式|18~20|
|单独前向填充|18~20|
|单独提高采样比例|16~19|
|单独调整pti|17~20|
|前向填充&提高采样比|21~23.5|
|前向填充&调整pti|19~21|
|提高采样比&调整pti|18~20|
|三者组合|均分25以上|

# 特征工程
我们做了特征工程来扩充数据，但并没有显著提高分数

- 滑动窗口统计--以1小时、6小时、24小时来进行滑窗，统计总和、均值、标准差、最大值、最小值
- 时间点--分钟、小时、天数
- 滞后时间窗口--以1、6、12、24小时；以1、5、15、30分钟进行统计
- 一阶差分--对每一个特征进行一阶差分
- 扩展窗口--基本统计
- 时间衰减--衰减因子为0.3
- 时间变化率--耗时长，内存大，最终未使用

# 其它实现
- RF与LGBM简单平均融合


# 持续改进与优化建议

|       主要内容        | 改进方式 |原因|
| :-----------------: |------|------|
|降低时间聚粒度|初始为5，可降低到2|内存未来是否故障和最近是否发生过故障似乎有很大关系，应更加关注短期趋势|
|特征构建|滑动窗口、滞后时间窗口|前向填充提高了预测分数，这可能表明未来内存在是否会在短期内故障和之前发生故障与否有很大关系|
|超参数调优|如果提高了采样比例，使用分层交叉验证、AUC进行评估、划分数据集时不要随机划分|提高采样比例后数据集严重倾斜；该数据集的很多特征具有时间相关性|
|控制采样随机性|加入随机数种子|不同采样得到了不同分布，不控制这一变量后面的改进具有很大的误导性|




