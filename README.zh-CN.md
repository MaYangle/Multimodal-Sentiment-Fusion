# Multimodal Sentiment Fusion

[English](README.md) · [中文](README.zh-CN.md)

Multimodal Sentiment Fusion 是一个研究语音、视频和文本的多模态情感分析项目，关注模态信号
发生冲突时，各路证据应如何影响预测，并保留各数据集的原生任务和评估规则。

**当前状态：研究进行中。** 这里是项目的新 GitHub 首页。代码、配置和经审计
的实验记录正在整理，首个仓库版本不宣称已有验证成功的原创方法、最终测试
成绩或发布检查点。

## 研究范围

- 优先完成 CH-SIMS v2.0 闭环，再以 CMU-MOSEI 复验，分别保留原生分数及划分。
- 先建立文本、简单融合和序列基线，再验证新机制。
- 对不依赖真实标签的模态策略与融合归一化进行受控实验。
- 真实训练、三种子比较、消融及独立预测重算。
- 保留失败假设、作者数据缺失和复现边界的准确记录。

项目本地工作名为 **JevAffect**。Jev 提供设计启发，本项目不使用其权重，也
不声称复现其未公开训练配方。路由、注意力和蒸馏已有广泛先例；实现网络本身
不能证明原创性或性能提升。

## 数据与复现

作者渠道数据获取与完整性核验仍在进行。首个仓库版本不包含数据、第三方
编码器权重或实验检查点；可运行源码发布时将说明来源与使用条件。当前首页
不是完整基准复现成果。

## 贡献方向

后续欢迎参与编码器集成、原生评估、融合算法、对照实验复现及错误分析。
完整运行命令和贡献指南将随源码版本提供。

## 参考入口

- [CH-SIMS v2 作者仓库](https://github.com/thuiar/ch-sims-v2)
- [CH-SIMS v2 数据页](https://thuiar.github.io/sims.github.io/chsims)
- [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)
- [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
