import config
# 导入数据获取函数
from src.data_loader import get_dataloaders
# 导入 KNN 分类器类
from src.model import KNNClassifier
# 导入准确率计算函数
from src.metrics import calculate_accuracy

# 定义主函数
def main():
    print("loading data\n")
    # 调用数据获取函数，传入配置参数，获取 6 个数据张量
    X_train, y_train, X_val, y_val, _, _ = get_dataloaders(
        test_size=config.TEST_SIZE, # 传入测试集比例
        val_size=config.VAL_SIZE,   # 传入验证集比例
        seed=config.RANDOM_SEED     # 传入随机种子
    )

    # 初始化最佳 k 值为第一个候选值
    best_k = config.K_CANDIDATES[0]
    # 初始化历史最高准确率为 0.0
    best_acc = 0.0

    print("find the best K\n")
    # 遍历配置文件中的每一个候选 k 值
    for k in config.K_CANDIDATES:
        # 使用当前的 k 值实例化 KNN 模型
        model = KNNClassifier(k=k)
        # 将训练数据喂给模型进行训练
        model.fit(X_train, y_train)
        # 使用模型对验证集进行预测
        predictions = model.predict(X_val)
        # 计算当前 k 值在验证集上的准确率
        acc = calculate_accuracy(predictions, y_val)
        
        # 打印当前 k 值及其对应的准确率
        print(f"当 K={k} 时，验证集准确率为: {acc * 100:.2f}%")
        
        # 如果当前准确率超过了历史最高记录
        if acc > best_acc:
            # 更新历史最高准确率
            best_acc = acc
            # 记录下这个表现最好的 k 值
            best_k = k
            
    print(f"\n最佳 K 值为 {best_k}，生成模型")
    
    # 使用刚刚找到的最佳 k 值，重新实例化一个最终的模型
    final_model = KNNClassifier(k=best_k)
    # 将训练数据喂给最终模型
    final_model.fit(X_train, y_train)
    # 调用模型的保存方法，将数据和最佳 k 值保存到配置文件指定的路径中
    final_model.save_model(config.MODEL_SAVE_PATH)

    print(f"最终模型保存至：{config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()