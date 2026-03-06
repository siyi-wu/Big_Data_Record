import os

# 定义随机种子，保证每次运行划分的数据集都是一样的，方便复现结果
RANDOM_SEED = 7 # 42
# 定义测试集占总体数据的比例（20%）
TEST_SIZE = 0.2
# 定义验证集占剩余训练数据的比例（25%，即总体的 20%）
VAL_SIZE = 0.25
# 定义 KNN 算法中 K 值的候选列表
K_CANDIDATES = [1, 3, 5, 7, 9]

# 定义模型保存的文件夹路径
MODEL_DIR = "./models"
# 定义最终模型保存的完整文件路径
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "knn_model.pth")

# 检查模型保存的文件夹是否存在，如果不存在则自动创建
if not os.path.exists(MODEL_DIR):
    # 创建 models 文件夹
    os.makedirs(MODEL_DIR)