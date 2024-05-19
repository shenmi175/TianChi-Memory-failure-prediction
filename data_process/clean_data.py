import re

# 定义一个函数来清理列名
def clean_column_names(column):
    # 保留中文字符、字母、数字和下划线
    return re.sub(r'[^\w\u4e00-\u9fff]', '', column)