# 数据驱动的质谱成像模式挖掘策略

[![License](https://img.shields.io/badge/License-3D%20Slicer%20Software%20License-green)](链接到你的协议页面)
![version](https://img.shields.io/badge/version-v2.1.2-blue)
![Dependency](https://img.shields.io/badge/dependency-PyTorch-orange)
![Language](https://img.shields.io/badge/language-Python-blue)

基于 Transformer 自编码器的质谱成像（3D MSI）数据模式挖掘工具。该项目通过深度学习模型对结直肠癌、胶质母细胞瘤（GBM）及前列腺癌的质谱成像数据进行无监督特征学习、重建和聚类分析。

---

## 核心方法

模型（`Computational_Model_trans.py`）采用 Transformer 编码器作为骨干网络，并引入三个关键机制：

| 机制 | 描述 |
|------|------|
| **Transformer 自编码器** | 多层 Transformer Encoder 学习质谱数据的全局上下文依赖，将高维质谱（m/z）压缩至低维隐空间表示 |
| **分数阶非局部记忆 (FracRNN)** | 通过分数阶权重矩阵对隐变量进行非局部时间聚合，增强模型对长程空间相关性的建模能力 |
| **受控噪声注入** | 在输入层和隐层注入自适应噪声（强度与数据标准差耦合），提升模型的鲁棒性和泛化能力 |

---

## 项目结构

```
.
├── Computational_Model_trans.py  # 核心模型定义（Transformer + FracRNN + 噪声机制）
├── utils.py                      # 矩阵运算工具函数
├── Fomae_3DColorectal_trans.ipynb # 3D 结直肠癌 MSI 分析流程
├── Fomae_GBM_trans.ipynb         # 3D 胶质母细胞瘤 MSI 分析流程
├── Fomae_prostate_trans.ipynb    # 3D 前列腺癌 MSI 分析流程
├── data/                              # 质谱成像数据集
├── Saved_models/                      # 训练好的模型权重
└── pic/                               # 可视化输出
```

---

## 环境依赖

- Python 3.7+
- PyTorch ≥ 1.10
- NumPy, SciPy
- h5py
- scikit-learn
- Matplotlib

