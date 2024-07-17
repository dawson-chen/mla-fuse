# MLA融合推理

MLA的设计可以保证效果与传统的MHA效果相同的情况下，实现更低的kv-cache开销。但是官方并没有给出矩阵融合后的推理代码，这对于对齐论文中的效果是必要的一步。

> 具体可以参考下面这个页面的讨论：
> 
> https://huggingface.co/deepseek-ai/DeepSeek-V2/discussions/1#6639bf1d01eaf0ea6fbf5e02

本仓库的代码用来实现MLA的推理，改动有以下2部分：

1. `fuse_mla_ckpt.py` 文件用来处理官方提供的ckpt文件，将其中MLA参数转换为融合后的参数；

2. `modeling_deepseek.py` 文件用来替换原来的推理脚本，加载融合后的ckpt进行推理；
