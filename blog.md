
## MLA计算过程

回顾一下MLA的计算过程，我们只看一个头的情况，因为多头计算只是单头结果的累加。

![image-20240709173442030](C:\Users\11103351\AppData\Roaming\Typora\typora-user-images\image-20240709173442030.png)
$$
q_i=W_{UQ} \cdot W_{DQ} \cdot h_i \\
\mathbf k=W_{UK} \cdot W_{DKV} \cdot \mathbf h \\
\mathbf v=W_{UV} \cdot W_{DKV} \cdot \mathbf h
$$

这三行是q，k，v的计算，因为attention会看上下文的信息，所以kv的计算需要用到上下文，用粗体h表示；带下标的h表示第i个token对应的隐层embedding向量；

下面是attention score的计算：

$$
\mathbf a_{i}=softmax(q_i^T \cdot \mathbf k) = softmax(h_i^T \cdot W_{DQ}^T \cdot W_{UQ}^T \cdot W_{UK} \cdot W_{DKV} \cdot \mathbf h)
$$

得到的a下标是一个形状为 [1, seq_len] 的行向量，在v [head_dim, seq_len] 向量上加权得到新的embeding后，通过O矩阵得到最后的输出。

$$
\hat h_i=W_{O} \cdot \mathbf v \cdot {\mathbf a_i}^T\\=W_{O} \cdot W_{UV} \cdot W_{DKV} \cdot \mathbf h \cdot softmax(h_i^T \cdot W_{DQ}^T \cdot W_{UQ}^T \cdot W_{UK} \cdot W_{DKV} \cdot \mathbf h)^T
$$

以上就是MLA nope头的计算方式，因为rope头的计算和MHA完全相同，并且不可以进行融合，所以不在这里考虑。

以Deepseek v2的结构设置为例，单个头计算涉及的矩阵形状如下：

$$
W_{DQ} \in \mathbb R ^ {1536 \times 5120} \\
W_{UQ} \in \mathbb R ^ {1536 \times 128} \\
W_{DKV} \in \mathbb R ^ {512 \times 5120} \\
W_{UK}, W_{UV} \in \mathbb R ^ {128 \times 512} \\
W_O \in \mathbb R ^{5120 \times 128}
$$

## GQA vs MLA vs MHA

我们按照deepseek v2的MLA参数，并按照常规设计GQA的思路，从hidden size推算出GQA的参数设置，得到GQA和MLA的2组参数设置如下：

|              |            MLA            | GQA  | MHA  |
| :----------: | :-----------------------: | :--: | :--: |
| hidden size  |           5120            |      |      |
|   head_dim   | 128 for nope, 64 for rope | 128  | 128  |
|  num_heads   |            128            |  40  |  40  |
| num_kv_heads |             -             |  4   |  40  |
| q_lora_rank  |           1536            |  -   |  -   |
| kv_lora_rank |            512            |  -   |  -   |

从表中可以看出，deepseek v2在MLA的头数上进行大量的补偿，并且每个头增加1/2维度用来携带位置编码信息，所以尽管用lora但是整体参数量相较于GQA仍然是增加的。

下面进行详细的计算3者的参数量：

Deepseek v2 MLA对比GQA

1. GQA: 
   - QKV=(40+4*2)*128*5120=31,457,280;
   - O=5120*5120=26,214,400;
   - total=57,671,680;
2. MLA: 
   1. 训练时候：
      - Q=5120\*1536 + 1536\*(128+64)\*128=45,613,056;
      - KV=5120\*(512+64) + 512\*(128+128)\*128=16,777,216;
      - O=5120\*128\*128=83,886,080;
      - total=146,276,352;
   2. 推理时，融合QK和OV参数；
      - QK_fuse=5120\*1536 + 1536\*(512+64)\*128=121,110,528;
      - KV=5120*(512+64)=2,949,120;
      - OV_fuse=5120\*512\*128=335,544,320;
      - total=459,603,968;
3. MHA:
   - QKV=5120*128*40\*3=78,643,200;
   - O=26,214,400;
   - total=104,857,600;

总结：MLA的参数是GQA的3倍，融合后是9倍，论文里对头数做了大量的补偿，从40增加到128，所以导致attn参数大幅增加，但是增加头数的原因不能简单看作是对MLA性能表现的补偿，所以不能理解为MLA相对于GQA存在参数劣势。

下面是假设在一个3B模型上使用MLA，并做和论文相同的参数补偿时，对应的参数量变化和kv cache的变化。

|          | 3B-GQA       | 3B-MLA | 3B-MLA-fuse | 3B-MHA       |
| -------- | ------------ | ------ | ----------- | ------------ |
| 参数量   | 2.46B        | 3.15B  | 3.9B        | 2.8B         |
| KV cache | 1024 / token | -      | 320 / token | 5120 / token |

至于如何在已经训练好的模型上快速进行MLA替换GQA，并且不损失效果，是后续需要考虑的点。

## 参数融合

*代码仓库 [chatgpt / MLA-fuse · GitLab (vmic.xyz)](https://gitlab.vmic.xyz/chatgpt/mla-fuse)*

MLA的设计可以保证效果与传统的MHA效果相同的情况下，实现更低的kv-cache开销。但是官方并没有给出矩阵融合后的推理代码，这对于对齐论文中的效果是必要的一步。

> 具体可以参考下面这个页面的讨论：
>
> https://huggingface.co/deepseek-ai/DeepSeek-V2/discussions/1#6639bf1d01eaf0ea6fbf5e02

本仓库的代码用来实现MLA的推理，改动有以下2部分：

1. `fuse_mla_ckpt.py` 文件用来处理官方提供的ckpt文件，将其中MLA参数转换为融合后的参数；
2. `modeling_deepseek.py` 文件用来替换原来的推理脚本，加载融合后的ckpt进行推理；

融合方式参考了论文中的描述，目前融合后计算是在pytorch上面进行的，可以结合ft做进一步的推理加速。



解释：为什么参数融合后，参数量反而会增加？

- QK的融合从[1536 x 128]和[512 x 128] 2个矩阵变成了一个 [1536 x 512]矩阵，从262,144增加到786,432；
- OV的融合从[128 x 512]和[5120 x 128] 2个矩阵变成了一个 [5120 x 512]矩阵，从720,896增加到2,621,440；
