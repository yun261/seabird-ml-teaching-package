# PPT 教学内容与项目文件对应表

| PPT 内容 | 对应材料 |
|---|---|
| 三轴加速度、25 Hz、2 秒窗口 | `data/teaching_samples/samples_acceleration.csv`、`samples_metadata.csv` |
| 原始数据预处理与滑窗 | `code/src/data_preprocess_logbot.py`、`code/notebooks/1data_preparation.ipynb` |
| 119 个手工特征 | `samples_features_119.csv`、`code/src/feature_extraction.py` |
| 静态/动态加速度 | `code/notebooks/7run_rolling_calc.ipynb` |
| 特征提取过程 | `code/notebooks/8run_feature_extraction.ipynb` |
| XGBoost 模型 | `models/xgboost/*.pickle`、`code/notebooks/9run_tree_model_training_and_test.ipynb` |
| XGBoost 特征重要性 | `results/xgboost/feature_importance*.csv`、`top20_feature_importance*.png` |
| DCL-SA 结构与推理 | `code/src/models/dcl_sa.py`、`models/dcl_sa/<animal_id>/` |
| DCL-SA 测试 | `code/notebooks/6run_test.ipynb` |
| Accuracy、Macro/Weighted F1、Macro IoU | 两个 `results/*/test_score*.csv` |
| 混淆矩阵所需真实/预测标签 | 两个 `results/*/y_gt_y_pred*.csv` |
| 行为数量与基础可视化 | `code/notebooks/simple_behavior_visualization.ipynb` |

## 六类教学标签映射

| 教学类别 | 原始行为标签 | 模型整数标签 |
|---|---|---:|
| Stationary | `stationary`、`preening` | 0 |
| Bathing | `bathing` | 1 |
| Take-off | `flight_take_off` | 2 |
| Cruising Flight | `flight_cruising` | 3 |
| Foraging Dive | `foraging_dive` | 4 |
| Dipping | `surface_seizing` | 5 |

注意：原始数据中的 `preening` 在该六分类实验中合并为 Stationary，`surface_seizing` 映射为 Dipping。这一映射来自项目现有特征数据和标签转换逻辑，不是教学包新定义的标签。

## 样本与模型对应关系

教学样本来自 OM1901、OM2211、OM2214。每个个体均配套：

- 以该个体为测试集的 XGBoost `.pickle`；
- 以该个体为测试集的 DCL-SA `best_model_weights.pt`；
- DCL-SA 对应 `config.yaml`。

这样可避免把目标样本所属个体同时放入训练数据，符合 PPT 中 LOIO-CV 的教学说明。

