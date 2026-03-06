import torch
# 导入数据集加载函数
from sklearn.datasets import load_iris
# 导入 sklearn 中的数据集划分函数
from sklearn.model_selection import train_test_split

# 定义获取数据加载器的函数，接收配置参数
def get_dataloaders(test_size, val_size, seed):
    # 调用函数，加载完整的 IRIS 数据集
    iris = load_iris()
    # 将原始特征数据转换为 PyTorch 的 32 位浮点数张量
    X = torch.tensor(iris.data, dtype=torch.float32)
    # 将原始标签数据转换为 PyTorch 的长整型张量
    y = torch.tensor(iris.target, dtype=torch.long)

    # 分离出测试集 (X_test, y_test) 和临时集 (X_temp, y_temp)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    
    # 从临时集中分离出验证集 (X_val, y_val) 和最终的训练集 (X_train, y_train)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=seed)

    # 将划分好的 6 个张量作为元组返回
    return X_train, y_train, X_val, y_val, X_test, y_test