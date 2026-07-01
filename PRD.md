# 图像生成方法总览 PRD

## 目标

本仓库当前处于规划阶段，目标是系统化整理并逐步实现多种图像生成方法，优先保证：

- 尽量使用成熟库、成熟模型类和预训练权重，而不是手写核心模型
- 优先兼容 Apple Silicon `MPS`
- 代码结构按方法拆分，便于独立实验和横向对比
- 当前主路线为小型 `Stable Diffusion` 风格流程

核心工程原则：

- 能用 `diffusers`、`accelerate`、`torchvision`、`PyTorch` 原生成熟组件时，不自行实现核心网络和训练基础设施
- 自己写的代码主要负责数据适配、路径管理、训练入口、日志、checkpoint 组织和不同模块之间的 glue code
- 不把“能跑通 toy demo”作为目标，优先降低训练不收敛、调参成本过高和后续接口推倒重来的风险
- 如确实必须自定义实现，必须先说明没有合适库的原因、风险和替代方案

## 当前阶段范围

当前已从纯规划进入第一阶段代码骨架搭建：

- 输出总览 `PRD.md`
- 创建后续实验所需目录结构
- 优先实现 `VAE` / latent 接口相关代码
- 不主动下载数据
- 不主动运行真实训练
- 允许进行不写文件的最小环境验证，例如检查 `MPS` 是否可用、随机 tensor 前向形状是否符合预期

## 方法范围

当前纳入统一规划的方法如下：

1. `VAE`
2. `GAN`
3. `DDPM`
4. `Flow Matching`
5. `DiT`
6. `Stable Diffusion`

## 总体方法判断

### VAE

- 是 `Stable Diffusion` 路线中的前置模块
- 不是本仓库优先自研的对象
- 当前主线直接使用 Hugging Face Diffusers 的 `AutoencoderKL`
- 默认优先加载预训练 `stabilityai/sd-vae-ft-mse`
- 如果后续确实需要适配 DAF 数据，也是在 Diffusers `AutoencoderKL` 基础上微调，而不是维护自定义 VAE 架构

### GAN

- 适合做生成基线
- 采样速度快
- 训练稳定性通常弱于扩散类方法

### DDPM

- 是扩散模型标准起点
- 方法成熟，资料多，库支持好
- 适合作为 diffusion baseline

### Flow Matching

- 可以视为连续时间生成建模的重要路线
- 值得纳入和 `DDPM` 的对照实验
- 后续实现时优先考虑基于现成库或已有训练框架封装

### DiT

- 适合做 transformer-based diffusion 路线验证
- 但对小数据、小算力场景不一定是第一优先级
- 在本仓库中先作为规划对象，不作为第一实现顺序

### Stable Diffusion

- 当前主路线
- 目标不是完整复刻工业级 SD，而是做一个小型、可训练、可解释、结构清晰的流程版本

## Stable Diffusion 主路线

本仓库后续优先采用如下流程：

1. 准备动漫头像数据集
2. 使用预训练或微调后的 Diffusers `AutoencoderKL`
3. 将 `128x128` 图像映射到 `4x16x16` latent 空间
4. 在 latent 空间训练扩散模型
5. 用 `VAE decoder` 将生成的 latent 解码回图像

## 为什么先确定 VAE

小型 `Stable Diffusion` 的核心是 latent diffusion，而不是直接在像素空间做扩散。因此：

- `VAE` 是前置模块
- 先有稳定的 encoder / decoder 和 latent 接口，后续 latent diffusion 才有意义
- 后续程序都会吃 `VAE encoder` 产生的 latent，因此 VAE latent 形状是系统级接口，不能随意改动
- 当前固定使用 `128x128` 输入，对应 latent 形状为 `[batch, 4, 16, 16]`
- 该接口与 Stable Diffusion 常用的 `latent_channels=4`、空间下采样 `8` 倍保持一致

当前默认 VAE 方案：

- 模型类：`diffusers.AutoencoderKL`
- 默认权重：`stabilityai/sd-vae-ft-mse`
- 输入范围：`[-1, 1]`
- 输入形状：`[batch, 3, 128, 128]`
- latent 形状：`[batch, 4, 16, 16]`
- latent scaling：使用 `vae.config.scaling_factor`，默认 `0.18215`
- 保存格式：优先使用 Diffusers `save_pretrained()` / `from_pretrained()`

## VAE 损失选择

如果需要微调 VAE，当前规划结论：

- 第一版优先使用 `MSE + KL`
- 不优先使用纯 `SSIM`

原因：

- `MSE` 更稳，更标准，库支持更成熟
- 首阶段更看重 latent 空间稳定性，而不是单纯追求视觉锐度
- `SSIM` 可以作为后续增强项，但不作为当前第一版主损失

当前默认方案：

- Reconstruction Loss: `MSE`
- Regularization Loss: `KL`
- 训练对象：Diffusers `AutoencoderKL`，不是自定义 VAE

后续可选增强方向：

- `L1 + KL`
- `MSE + alpha * SSIM + KL`

## 数据集方向

当前主路线数据集目标：

- 动漫头像
- 分辨率固定优先 `128x128`
- 面向小型 `Stable Diffusion`

数据源策略：

- 优先使用 `Danbooru` / `Danbooru2021` 作为上游来源
- 不优先直接采用来路不明的二次压缩 `64x64 anime face zip`
- 后续在 `dataset/` 下整理筛选规则、清洗策略、采样标准

原因：

- `Danbooru` 标签体系更完整
- 可控性高
- 更容易过滤掉低质量、多人图、奇怪构图、歪脸和杂质样本

## 目录规划

```text
dataset/
vae/
gan/
ddpm/
flow_matching/
dit/
stable_diffusion/
```

## 计划中的职责划分

### dataset/

- 放数据集说明
- 放数据筛选规则
- 放后续下载、清洗、划分方案

### vae/

- 放 `VAE` 训练与重建实验
- 作为 `Stable Diffusion` 前置模块

### stable_diffusion/

- 放 latent diffusion 主流程实现
- 后续依赖 `vae/` 产出的 encoder / decoder

### 其他方法目录

- `gan/`：经典生成基线
- `ddpm/`：扩散基线
- `flow_matching/`：连续时间路线
- `dit/`：transformer diffusion 路线

## 实现约束

- 框架优先 `PyTorch`
- 优先选用成熟库、现成模型类和预训练权重
- 尽量避免手写底层训练基础设施和核心生成模型结构
- 能调库绝不手写
- 默认考虑 `MPS`
- 若某库对 `MPS` 兼容差，则优先选择更稳的替代方案

当前优先库：

- `diffusers`：`AutoencoderKL`、`UNet2DModel`、scheduler、后续 pipeline 组件
- `accelerate`：后续训练加速、设备管理和混合精度能力
- `torchvision`：图片读取、基础 transform、样例图保存
- `PyTorch` 原生组件：optimizer、scheduler、DataLoader

明确不优先做的事情：

- 不手写 VAE / U-Net / scheduler 的核心算法，除非成熟库不能满足最小目标
- 不为了教学完整性牺牲训练稳定性
- 不把随机初始化的大模型训练作为默认方案
- 不默认从零训练 VAE；默认复用预训练 SD VAE，必要时再微调

## 训练控制约束

- 所有方法后续实现时默认都要支持 `Early Stopping`
- 优先采用成熟训练框架或成熟工具库提供的 `Early Stopping` 能力
- 不优先手写早停逻辑，除非库能力无法满足需求
- 所有方法后续实现时默认都要接入成熟的学习率调度机制
- 学习率调度器优先使用现成库实现，不手写 scheduler

推荐原则：

- 能直接使用框架回调或现成组件时，不重复造轮子
- 优先选择稳定、常用、文档完善、对 `MPS` 兼容较好的方案
- 在效果接近的前提下，优先选工程维护成本更低的方案

后续实现阶段可优先考虑的方向：

- `PyTorch` 原生 scheduler
- `Lightning` 等训练框架提供的 `Early Stopping` / checkpoint / logging 机制
- `diffusers`、`timm`、`accelerate` 等成熟生态中的现成训练组件

## 代码注释规范

后续所有代码实现都必须包含充分中文注释，不能只写最少量说明。

具体要求：

- 每个程序文件开头必须说明本文件的意图、用途、所处训练流程位置
- 每个核心函数都要写清楚作用、主要输入、输出和关键逻辑
- 重点逻辑必须写中文注释
- 较长、较绕、较难读懂的语句或代码块必须补中文说明
- 训练主流程、数据流向、模型输入输出形状变化等关键点必须写清楚
- 不是所有简单语句都要逐行注释，但不能把复杂代码裸放在那里不解释

注释目标：

- 让后续回看代码时能快速理解设计意图
- 让不同方法目录下的实现风格尽量统一
- 优先保证可维护性和可读性，而不是追求最少注释

## 当前结论

当前仓库后续主线不是“所有方法同时开工”，而是：

1. 先完成 `Stable Diffusion` 方向规划
2. 先固定 VAE latent 接口和 Diffusers `AutoencoderKL` 使用方式
3. 默认使用预训练 `stabilityai/sd-vae-ft-mse`，必要时再基于 `MSE + KL` 微调
4. 其余方法目录先占位，后续按优先级逐步实现
