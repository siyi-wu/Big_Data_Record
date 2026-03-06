# 定义计算准确率的函数，接收预测标签和真实标签
def calculate_accuracy(y_pred, y_true):
    # 比较预测值和真实值，生成布尔张量，求和得到正确预测的总个数，并提取出具体数值
    correct = (y_pred == y_true).sum().item()
    # 用正确个数除以样本总数，得到准确率的小数形式
    accuracy = correct / y_true.size(0)
    # 返回计算出的准确率
    return accuracy