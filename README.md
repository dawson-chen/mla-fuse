# mla-fuse
MLA的设计可以保证效果与传统的MHA效果相同的情况下，实现更低的kv-cache开销。但是官方并没有给出矩阵融合后的推理代码，这对于对齐论文中的效果是必要的一步。本仓库的代码用来实现MLA的参数融合，以及融合后的pytorch推理代码。