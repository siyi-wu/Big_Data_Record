import config
# 导入数据获取函数
from src.data_loader import get_dataloaders
# 导入 KNN 分类器类
from src.model import KNNClassifier
# 导入准确率计算函数
from src.metrics import calculate_accuracy

# 定义主函数
def main():
    print("获取测试集数据")
    # 调用数据加载函数。由于随机种子相同，这里获取到的测试集与 train.py 划分出的一样
    # 只接收最后两个返回值（测试集）
    _, _, _, _, X_test, y_test = get_dataloaders(
        test_size=config.TEST_SIZE, # 传入测试集比例
        val_size=config.VAL_SIZE,   # 传入验证集比例
        seed=config.RANDOM_SEED     # 传入一致的随机种子以确保数据一致性
    )

    print(f"从 {config.MODEL_SAVE_PATH} 加载模型")
    # 实例化一个空的 KNN 模型对象（不需要传 k 值，因为马上要从文件中读取）
    model = KNNClassifier()
    # 调用加载方法，读取 .pth 文件中的训练数据和最佳 k 值
    model.load_model(config.MODEL_SAVE_PATH)
    # 打印读取到的最佳 k 值
    print(f"记录的最佳 K 值为: {model.k}")

    print("\n对测试集进行预测")
    # 使用加载好的模型对测试集特征进行预测
    predictions = model.predict(X_test)
    # 计算预测结果在测试集上的最终准确率
    final_accuracy = calculate_accuracy(predictions, y_test)

    # 打印结果
    print("=" * 20)
    print(f"模型在测试集上的准确率为: {final_accuracy * 100:.2f}%")
    print("=" * 20)

if __name__ == "__main__":
    main()