运行指南如下：

一、代码依赖
(1)Python 版本：建议使用 Python 3.6 及以上版本。
(2)依赖库：
①numpy：数值计算。
②matplotlib.pyplot：数据可视化。
③scikit - learn：机器学习模型。


二、环境配置
(1)确保已安装 Python 3.6 及以上版本。
(2)使用 pip 安装依赖库：pip install numpy matplotlib scikit - learn


三、数据准备
(1)代码中使用 numpy 生成模拟数据，无需额外准备数据集。数据生成逻辑：
①disciple_count：通过 np.random.randint(50, 200, size=20) 随机生成 20 个 50 到 200 之间的整数，表示弟子数量。
②master_skill：通过公式 3 * disciple_count + np.random.randn(20) * 20 + 100 生成，模拟掌门武功修为，部分代码中对超过 600 的值进行了截断处理（np.where(master_skill > 600, 600, master_skill)）。


四.运行步骤
(1)代码文件保存
将每份代码分别保存为独立的 .py 文件，例如：
①第一份代码保存为 linear_regression_visual.py
②第二份代码保存为 linear_regression_preprocess.py
③第三份代码保存为 linear_vs_ridge_regression.py
④第四份代码保存为 linear_vs_ridge_regression_mae.py

(2)运行代码
①打开终端或命令提示符，导航到代码所在目录。
②分别运行每份代码，命令如下：
python linear_regression_visual.py
python linear_regression_preprocess.py
python linear_vs_ridge_regression.py
python linear_vs_ridge_regression_mae.py

(3)代码功能说明
①第一份代码（linear_regression_visual.py）：生成模拟数据，训练线性回归模型，绘制散点图、残差图、预测值与实际值对比图、残差直方图。
②第二份代码（linear_regression_preprocess.py）：生成模拟数据，进行数据预处理（异常值处理、标准化、划分训练集和测试集），训练线性回归模型，对比原始数据和标准化后数据的模型参数，并可视化展示。
③第三份代码（linear_vs_ridge_regression.py）：生成模拟数据，预处理后训练线性回归和岭回归模型，评估模型的 MSE 和 R2，并可视化对比原始数据分布、线性回归和岭回归的预测结果。
④第四份代码（linear_vs_ridge_regression_mae.py）：在第三份代码基础上，增加 MAE 评估指标，进行错误案例分析，输出两种模型误差最大的样本信息。


五.注意事项
(1)确保运行环境网络连接正常。
(2)如果可视化图表显示异常，检查 matplotlib 配置是否正确，或尝试重新安装 matplotlib。
(3)代码中设置了随机种子 np.random.seed(42)，确保每次运行生成的模拟数据一致，便于复现结果。
