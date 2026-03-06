import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 从 sklearn 导入鸢尾花数据集加载函数
from sklearn.datasets import load_iris


def main():

    print("加载数据，可视化")
    
    # 加载并整理数据
    # 加载完整的 IRIS 数据集
    iris = load_iris()
    # 将 NumPy 格式的特征数据转换为 Pandas 的 DataFrame 表格，并附上列名（4个特征的名称）
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # 在表格最后新增一列 'species'（品种）。
    # iris.target 里是 0, 1, 2，通过 iris.target_names 把它映射成具体的英文花名
    df['species'] = [iris.target_names[i] for i in iris.target]
    
    # 准备输出目录
    # 定义图片保存的文件夹相对路径
    output_dir = "./images"
    # 如果这个文件夹不存在
    if not os.path.exists(output_dir):
        # 就自动在当前目录下创建它
        os.makedirs(output_dir)
        
    # 绘图
    # 打印提示信息
    print("绘制特征散点图矩阵")
    sns.set_theme(style="whitegrid")
    # 调用 pairplot 函数，传入我们的数据表格。
    # hue='species' 表示根据 'species' 这一列的不同类别，使用不同的颜色区分散点
    # markers 设定了三种花分别对应 圆圈、方块、菱形 三种不同的散点形状
    sns.pairplot(df, hue='species', markers=["o", "s", "D"])
    
    # 保存图表
    # 拼接路径
    save_path = os.path.join(output_dir, "iris_features_pairplot.png")
    # 将刚才画好的图表保存为高清（dpi=300）的 PNG 图片，bbox_inches='tight' 防止图例被边缘裁剪
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"可视化完成，已保存至: {save_path}")

if __name__ == "__main__":
    main()