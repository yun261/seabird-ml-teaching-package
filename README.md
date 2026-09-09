##  快速体验

如果你只想运行教学程序，无需配置 Python 环境：

1. 进入 Releases
2. 下载最新版本的便携包 或者点击  https://github.com/yun261/seabird-ml-teaching-package/releases/download/v1.0.0/seabird_visualization_portable_windows.zip  直接下载
3. 解压 ZIP（内有详细步骤）
4. 双击 run_app.bat
5. 浏览器会自动打开教学页面



# 海鸟行为识别教学材料包说明

基于三轴加速度与机器学习的海鸟行为识别交互教学工具。

本文件夹用于配合课程 PPT、现有可视化演示。原项目没有被移动或修改；源码、配置、模型和结果主要为原样复制，教学数据由真实项目数据整理生成，不包含合成数据。

## 1. course_ppt：课堂讲解材料

- `讲解ppt0909.pptx`：课程主 PPT。
- 内容包括三轴加速度、25 Hz、2 秒窗口、六类行为、119 个特征、XGBoost、DCL-SA、LOIO-CV、评价指标和混淆矩阵。

## 2. data/visualization_inputs：现有可视化的真实输入

`visualization_inputs/omizunagidori` 完整保存原可视化 Notebook 实际读取的个体原始 CSV、`label_all2.csv` 和已有的 `visualization_results/`。

主要字段为 `timestamp, acc_x, acc_y, acc_z, label`。

用途：上传到现有便携可视化 Web，或供 `simple_behavior_visualization.ipynb` 等行为分布 Notebook 使用。

## 3. data/teaching_samples：完整统一教学数据

这部分服务后续教学型机器学习 Web，不是原始可视化 CSV 的重复副本。

当前覆盖 42,526 个真实窗口、28 个个体；每窗 50 点，采样率 25 Hz，时长约 2 秒。

### samples_metadata.csv

每行代表一个窗口，共 42,526 行。包含统一 `sample_id`、个体、教学类别、原始标签、起止时间、采样率、窗口点数和源文件路径。

### samples_acceleration.csv

保存全部窗口的三轴加速度，共 2,126,300 行。每个 `sample_id` 恰好对应 50 行。

来源：`D:/dl-wabc/data/datasets/preprocessed_data/omizunagidori/*.csv`。

### samples_features_119.csv

每行对应一个窗口，共 42,526 行。保存 XGBoost 使用的 119 个真实特征和原有元数据，只额外增加统一 `sample_id`。

来源：`D:/dl-wabc/data/datasets/logbot_data/feature_extraction/acc_features/omizunagidori/*.csv`。

### manifest.json

记录窗口数、加速度行数、个体与类别数量、来源目录、匹配规则和是否包含合成数据。

四个文件通过 `sample_id` 关联，采用 `animal_id + 窗口末端 unixtime` 对齐。

## 4. code/src：核心 Python 代码

这些是原项目源码副本，核心算法未修改。

- `data_preprocess_logbot.py`：时间戳、重采样、预处理、滑窗和 NPZ。
- `feature_extraction.py`：静态/动态加速度与手工特征。
- `data_module.py`：PyTorch Dataset、DataLoader 和数据划分。
- `trainer.py`：模型建立、训练、测试和保存。
- `utils.py`：标签、指标、混淆矩阵和统计图。
- `augmentations.py`：数据增强。
- `env.py`：随机种子。
- `gpu.py`：GPU 检测。
- `pytorchtools.py`：Early Stopping。
- `models/`：CNN、LSTM、DCL、DCL-SA、Transformer、CNN-AE 等模型结构。

## 5. code/notebooks：相关 Jupyter Notebook

- `1data_preparation.ipynb`：原始数据预处理和窗口生成。
- `6run_test.ipynb`：加载深度学习权重并测试。
- `7run_rolling_calc.ipynb`：静态/动态加速度计算。
- `8run_feature_extraction.ipynb`：119 特征提取。
- `9run_tree_model_training_and_test.ipynb`：树模型训练与测试。
- `simple_behavior_visualization.ipynb`：行为分布可视化。

这些 Notebook 默认使用原项目路径，不会自动读取 `teaching_samples`，需要手动进行修改。

## 6. code 下的数据整理脚本

- `build_complete_teaching_data.py`：生成当前完整四个教学数据文件。
- `build_teaching_samples.py`：早期 60 窗口版本的构建记录，当前完整数据不由它提供。
- `build_visualization_inputs.py`：生成小型可视化输入的辅助脚本。

日常使用不需要运行这些脚本。

## 7. configs：实验配置

包括深度学习、树模型、CNN-AE、数据集、25/78/119 特征集合、模型结构和路径配置。这些是原项目配置副本。可以修改学习率、epoch、batch size、CUDA 等训练参数。

## 8. models/xgboost：已有 XGBoost 模型

包含 `OM1901.pickle`、`OM2211.pickle`、`OM2214.pickle` 三个 LOIO 个体模型。

## 9. models/dcl_sa：已有 DCL-SA 模型

包含 OM1901、OM2211、OM2214 三组 `best_model_weights.pt` 和 `config.yaml`。

## 10. results/xgboost：XGBoost 真实实验结果

包含测试指标、真实/预测标签、全局特征重要性、平均重要性排名和 Top 20 特征重要性图片。用于评价指标、混淆矩阵及“模型主要参考什么”的教学展示。

## 11. results/dcl_sa：DCL-SA 真实实验结果

包含 DCL-SA 测试指标和真实/预测标签，用于生成真实混淆矩阵并与 XGBoost 比较。

## 12. docs

- `FILE_MAP.md`：PPT 内容与代码、数据、模型及结果的快速对应表。

## 13. 应该使用哪一部分

现有可视化 Web 或行为分布 Notebook使用：`data/visualization_inputs/omizunagidori/*.csv`。

## 14. 注意事项

- 不用重新训练模型。
- 不要让 Web 每次读取原项目完整 23 GB 数据。
- 全局特征重要性不等于单一样本的因果解释。
- 原始 Notebook、源码和模型均保持原来的运行方式。
