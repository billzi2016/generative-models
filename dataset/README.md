# dataset

## 目录用途

本目录用于存放当前项目的数据集文件、数据说明以及后续的数据整理约定。

当前主路线围绕小型 `Stable Diffusion` 流程展开，因此数据集选择优先服务于：

- 动漫头像生成
- `VAE -> latent diffusion` 流程
- 优先保留较高质量的人脸/头部样本

## 当前选定数据集

当前优先使用 `DAF:re / DAFB` 路线的数据。

选择原因：

- 是别人已经整理过的动漫头像/头部数据，不是随意打包的小型民间压缩包
- 来源与 `Danbooru` 路线相关，整体质量和一致性通常优于杂乱来源
- 数量足够大，适合后续 `VAE` 和 latent diffusion 实验

## 当前已确认的信息

- 数据集名称：`DAFB`
- 上游背景：`DAF:re (DanbooruAnimeFaces:revamped)` 相关路线
- 图像分辨率：`128x128`
- 数据规模：
  - 论文中的 `DAF:re Faces` 最终版本约 `463,437` 张图
  - `Hugging Face` 上当前公开包名为 `daf.tar.gz`
- `Hugging Face` 数据集页显示占用约 `13.34 GB`

## 当前下载策略

当前不自己从全量 `Danbooru2021` 清洗，而是优先下载现成整理好的数据包。

当前实际数据位置：

- `raw/fullMin256/`
- `raw/train.csv`
- `raw/train_val.csv`
- `raw/classid_classname.csv`

这样做的原因：

- 下载成本更可控
- 可以尽快进入后续 `VAE` 与 `Stable Diffusion` 主流程
- 先避免把时间耗在上游大规模清洗上

## 当前目录约定

```text
dataset/
  README.md
  raw/
    README.md
    classid_classname.csv
    train.csv
    train_val.csv
    fullMin256/
```

说明：

- `raw/` 当前直接保存已解压后的 DAF 数据
- `raw/README.md` 是上游数据集自带说明，保留进仓库；图片、CSV 等大体积数据仍作为本地数据文件处理
- `raw/fullMin256/` 是当前 VAE 默认读取的图片目录
- 训练代码会递归扫描 `raw/fullMin256/` 下的图片文件，不依赖单层目录结构

## 与训练分辨率的关系

当前主线固定使用 `128x128` 作为 VAE 输入分辨率。

当前建议是：

- 原始数据保留在 `dataset/raw/fullMin256/`
- VAE 训练阶段统一 resize 到 `128x128`
- 后续 latent 形状固定为 `[batch, 4, 16, 16]`

## 后续补充内容

后面会继续在本目录补充：

- 解压方式
- 训练/验证划分策略
- 是否需要进一步过滤异常样本
