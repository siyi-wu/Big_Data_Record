import torch
# 导入 Counter 类，用于进行多数表决时的计数
from collections import Counter

# 定义 KNN 分类器类
class KNNClassifier:
    # 初始化，接收参数 k
    def __init__(self, k=3):
        # 将传入的 k 值保存为类的实例属性
        self.k = k
        # 初始化训练特征属性为 None
        self.X_train = None
        # 初始化训练标签属性为 None
        self.y_train = None

    # 定义训练方法，接收训练特征和标签
    def fit(self, X_train, y_train):
        # 将训练特征存储在模型中
        self.X_train = X_train
        # 将训练标签存储在模型中
        self.y_train = y_train

    # 定义预测方法，接收测试/验证特征
    def predict(self, X_query):
        # 初始化一个空列表保存所有的预测结果
        predictions = []
        # 逐个遍历传入的待预测样本
        for point in X_query:
            # 计算当前待测样本与所有训练样本的欧氏距离
            distances = torch.sqrt(torch.sum((self.X_train - point) ** 2, dim=1))
            # 找到距离最近的 k 个样本的索引 (largest=False 表示找最小值)
            _, indices = torch.topk(distances, self.k, largest=False)
            # 根据索引，提取出这 k 个最近样本的真实标签，并转为 Python 列表
            k_labels = self.y_train[indices].tolist()
            # 统计这 k 个标签中出现次数最多的那个类别
            most_common = Counter(k_labels).most_common(1)[0][0]
            # 将得票最多的类别加入到预测结果列表中
            predictions.append(most_common)
        # 将包含所有预测结果的列表转换为 PyTorch 张量并返回
        return torch.tensor(predictions)

    # 定义保存模型的方法
    def save_model(self, filepath):
        # 将训练特征、标签和 k 值打包成一个字典，序列化保存为 .pth 文件
        torch.save({'X_train': self.X_train, 'y_train': self.y_train, 'k': self.k}, filepath)

    # 定义加载模型的方法
    def load_model(self, filepath):
        # 从指定的 .pth 文件中反序列化读取数据字典
        checkpoint = torch.load(filepath)
        # 恢复训练特征属性
        self.X_train = checkpoint['X_train']
        # 恢复训练标签属性
        self.y_train = checkpoint['y_train']
        # 恢复 k 值属性
        self.k = checkpoint['k']