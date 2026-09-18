# 第五批落地记录：自主 latent 执行（20 runs）

日期：2026-09-18。方案：[experiments_latent_batch2_20runs.md](experiments_latent_batch2_20runs.md)（不改动）；清单：[manifest](latent_batch2_20runs_manifest.json)。本文记录实现、核查、偏差和作业。

## 1. 作业

| 作业 | 内容 | walltime | 状态 |
|---|---|---|---|
| **3391720–3391739 lat2-01 … lat2-20** | **每个 run 独占一个节点**（run id = 作业名后两位，顺序同 `configs/cells_latent2.txt`），无 MPS，记在 **gj26** 项目下（`-W group_list=gj26`；用户 09-18 21:45 批准使用 gj26 并要求一节点一任务）。单进程 48.5 ms / update → 250k ≈ 3.4 h，加监测与 final 评估约 3.6 h | 5 h | 排队 09-18 21:46 |
| 3391711 lat2-a | run 01 / 05 / 08 / 11 / 15（A–E 的 seed 1）+ 02 / 06 / 09 / 12 / 16（A–E 的 seed 7），10 个进程同卡，`MPS=1` | 6.5 h | 开跑前撤回（被一节点一任务取代） |
| 3391712 lat2-b | run 03 / 07 / 10 / 13 / 17（A–E 的 seed 123）+ 04 / 14（A、D 的 seed 2026）+ 18 / 19 / 20（F × 3），10 个进程同卡，`MPS=1` | 6.5 h | 开跑前撤回（被一节点一任务取代） |
| 3391572–3391575 | 最初的 4 × 5 打包（无 MPS，4 h），在开跑前撤回，被上面两个作业取代 | — | 已 qdel |
| 3391569 lat2-tput | 吞吐测试：10 进程同卡、无 MPS，各 1,200 updates + 同时跑一次完整 final 评估 | 25 min | 完成 09-18 21:22 |
| 3391702 lat2-mps / 3391703 lat2-solo | 同样 10 进程但开 NVIDIA MPS；以及单进程独占一卡 | 15 / 10 min | 完成 09-18 21:31 |

**吞吐实测（GH200，每 update 的墙钟，含 1k / 1.2k 两次监测评估）：**

| 布局 | A–D、F | E（T=1） | 10 个 run 的总吞吐 |
|---|---|---|---|
| 单进程独占一卡 | 48.5 ms | — | 1×（基准） |
| 10 进程同卡，无 MPS | 139–141 ms | 98.8 ms | 3.4× |
| 10 进程同卡，MPS | 70.5–72.0 ms | 31.5 ms | 6.7× |

这一步极小（256 行 × ≤ 8 token、5M 参数），单进程受 Python / kernel 启动开销限制；不开 MPS 时各进程的 CUDA context 只能分时，开 MPS 后吞吐翻倍。按 72 ms 估，250k updates ≈ 5.0 h，加监测与 final 评估（10 进程同时评估时每个 A–D 约 600 s、E 25 s、F 50 s，峰值显存 4.0–4.2 GB / 进程）约 5.4 h；两个节点合计约 11–13 node-hours，而原先 4 × 5、无 MPS 的布局每个作业要 5 h 以上、需要续跑，总计约 24 node-hours。真实数据上的早期学习信号：答案 loss 在 300–800 updates 停在 6.23（= ln 500，实体随机水平），900 updates 后开始下降；小世界 CPU 试跑 w2 验证到 1.000。

20 个 run 每个都是 `train_latent.py --resume`：训练到 250k 后在同一进程内写 `final_eval.json`（幂等）；walltime 不够就用同一条 qsub 命令续跑，不改配方。go39 的 token 余额约 125 node-hours（`show_token`：203,040 中已用 202,914.7），两个 6.5 h 作业被接受；**之后任何一批都需要新的 token 或改用别的项目额度，这一点需要用户决定。**

## 2. 实现（方案 §11.1 的文件）

- `src/data/latent_format.py`：词表 = `chain_loop/vocab.json` 524 项 + `TASK EOP EMPTY ANSWER END_ANSWER` = 529；两个 131 位置的 TASK 区（TASK、起点、128 槽、EOP），逻辑地址 0–261；有序随机槽位；**packed 表示**（只物化有效 token，保留原始 position id）；训练 batch 由张量运算直接拼出（w2 8 token、d2 5 token）；辅助标签先记逻辑地址，再由 `addr_to_key` 按每个样本的 position id 映射到 key 下标；严格答案解析。
- `src/model/latent_executor.py`：2 层双向 Encoder → M（每次求解只算一次）；4 个 learned latent slot；core = 2 个 block（latent self-attn → 对完整 M 的 cross-attn → FFN，pre-LN，残差支路输出投影零初始化）；`shared`（A–D）/ `single`（E）/ `untied`（F：8 份独立 core，第 t 轮用第 t 份）；1 层因果 AnswerDecoder 只 cross-attend 最终 Z（`decode(z, prefix)` 没有 memory 参数）；RoPE base 100，Q/K 投影后旋转、V 不旋转，memory 用逻辑坐标、latent 262–265、答案前缀从 266 起；被监督读取 = 第二个 block 的 cross-attn、slot 0、head 0 的真实 soft 概率（该模块在**所有条件**都走手写 softmax 路径，其余用 SDPA）；实体辅助头（LN + linear → 500 类）在所有条件实例化；每个张量由 `(seed, 参数名)` 决定的独立 generator 初始化，所以 A–E 同名张量逐位相同，F 的公共部分与 A 相同。
- `src/train_latent.py`：方案 §5–§7 的损失、日程、监测、选点、检查点、最终评估；独立 RNG 流（来源 w2 / d2、布局、T），评估不碰训练流；阶段一完全不建辅助图。
- `src/generate_latent_eval.py`：冻结评估布局规则与 hash（`610700e0a5cedd6b`）、每格前 500 / 后 4500 的划分与不相交检查、500 对后缀干预题（d3 / 8 / 32 / 128 各 125 对）→ `data/chain_loop/latent_eval_manifest.json`。
- `scripts/test_latent_executor.py`、`scripts/pbs_latent.sh` + `configs/cells_latent2.txt`（20 行显式 run，不与 seed 列表做笛卡尔积）、`scripts/summarize_latent.py`（只读汇总：主结果 / 判据 / 因子效应 / 非法格式 / 未完成 run 的监测数，分开标注）。

参数量（词表 529、d=256，`manifest.json` 里有分项）：A–E 约 5.14M（Encoder 1.58M、core 2.11M、decoder 1.05M、embedding / 输出头各 0.135M、辅助头 0.128M、latents 1,024）；F 多 7 份 core，约 19.9M。

## 3. 方案 §10 的十项检查（`python scripts/test_latent_executor.py`，CPU，全部通过）

1. 数据：train / val / vocab 的 hash 与 C_k1 相同，40k w2 + 10k d2 + 0 d3；batch 构造器只接收 w2 / d2，训练 batch 里每个程序 ≤ 2 个关系；词表 = 原 524 项 + 5。
2. 接口：`forward(tok, pos, pad, T, ans_in, collect)`；改动 gold 辅助标签后所有前向输出逐位不变。
3. 循环：T=2 与 T=8 的 M 逐位相同；Z 每轮都变；答案 loss 的梯度到达第 1 轮的 Z、Z0 和 Encoder（无 detach / reset）。
4. 完整程序：d=128 的 packet 有 131 个 token、地址到 EOP；只改**最后一个**关系，M 与答案 logits 都变。
5. dense（262 位置 + EMPTY mask）= packed：logits、读取 loss、梯度一致；多余 PAD 列不影响任何输出。
6. A–D 同 seed：阶段一结束时参数 hash 完全相同，辅助头未被触碰且不在 AdamW state 中（不会被 weight decay）；进入阶段二后四者各不相同。
7. 参数：E 与 A 完全相等；F 有 8 份不共享 storage 的 core，公共张量与 A 逐位相同；F 各 core 的参与次数与 T 抽样一致（例 12 步：[12, 12, 8, 8, 5, 5, 5, 5]）。
8. Decoder 无法读 M；不同 teacher-forced 答案下 Z_T 逐位相同；生成为全词表 greedy。
9. 续跑：8 updates 直跑 = 4 + resume 4（参数 hash 相同；模型、AdamW、来源 / 布局 / T 流、计数器均恢复）。
10. 小 batch 的参数与耗时探针（GPU 吞吐由 3391569 实测）。

另：小世界（E=60、R=10、d=64）CPU 上 A / D 的 loss 8.6 → 3.4（600 updates，低于均匀猜测的 ln 60 = 4.1）且阶段一 A、D 的 loss 逐点相同。

## 4. 与方案的偏差（需要披露）

- **模型检查点**：完整续跑状态 `last.pt` 每 5k 覆盖保存；只含模型的 `ckpt_*.pt` 存 25k 的倍数 + 240k / 245k / 250k + `best.pt`（每 5k 都存会多占约 17 GB）。
- **最终评估与训练同作业**（幂等，可 `--eval_only` 重跑），没有另排评估作业：T=256 的全量评估很便宜，而 token 紧张；`--stop_after` + `POST_EVAL` 只用于吞吐测试。
- **布局的三次变化（都发生在开跑之前，训练代码与配方不变）**：4 × 5 无 MPS → 实测后 2 × 10 + MPS（省 go39 的 token）→ 用户批准 gj26 后改为 20 × 1（一节点一任务，最快，run 之间互不影响，也不依赖 MPS）。walltime 不够靠 `--resume` 续。
- **梯度量级**：只记全局（裁剪前）梯度范数与裁剪触发比例；辅助 loss 分别记"执行轮（t ≤ 2）/ 完成后轮（t ≥ 3）"的 CE，没有逐 loss 单独反传求梯度范数。
- **FLOPs** 为解析估计（线性层 + 注意力矩阵乘，训练 = 3 × 前向），不是实测。
- **唯一布局输入数**未逐个统计（布局空间远大于呈现次数，几乎每次呈现都是新输入）。
- 方案未规定的结构细节：Encoder 末尾一个 LayerNorm；Z 进 decoder cross-attn 前一个 LayerNorm；Encoder 与 decoder 共用输入 embedding，输出头不共享。
- 后缀干预题从 random 类别前 500 题构造，改动位置在 `[d//2, d)`，不引入验证邻接，两题 gold 不同，同一对共用一个布局。

## 5. 中期结果（09-19 04:00，前 5 个完成的 run；其余 15 个在跑或排队，最终表待全部完成后替换本节）

已完成：A 的 seed 7 / 123 / 2026，D 的 seed 123 / 2026（各 250k updates + 全量 final 评估，每个 run 约 3.6 h，final 评估 127–130 s）。

| 读数（250k final） | A_s7 | A_s123 | A_s2026 | D_s123 | D_s2026 |
|---|---|---|---|---|---|
| d2 训练关系对拟合（all_seen，T=8） | 0.997 | 0.999 | 0.999 | 0.998 | 1.000 |
| **d2 未见关系对（one_new / random 取低者，T=8）** | **0.002** | **0.002** | **0.002** | **0.003** | **0.002** |
| d3 – d128（所有类别，T=8 与 T=256） | ≤ 0.003 | ≤ 0.003 | ≤ 0.003 | ≤ 0.003 | ≤ 0.003 |
| d2 训练对拟合在 T=256 | 0.949 | 0.999 | 0.991 | 0.660 | 0.613 |
| w2 验证整行（新配对的已知事实） | 0.40 | 1.00 | 1.00 | 1.00 | 1.00 |
| **单任务原子题 `TASK e r EOP`（10,000 项，只评估）** | 0.011 | 0.024 | 0.054 | 0.031 | 0.004 |
| 240k / 245k / best 的 d2 未见对 | 0.004–0.008 | 0.002 | 0.002–0.004 | 0.004–0.008 | 0.002–0.004 |

- **预注册判据（§8.1）**：d2 门槛（未见对 ≥ 95%）0 / 5 通过，之后的深组合判据自然都不成立。同时给了实体与读取两种辅助监督的 D 与纯答案监督的 A 没有可见差别：未见关系对都在 chance（1/500 = 0.002）。
- **学到的是什么**：d2 训练对 100% 拟合、未见对为 chance，说明 10,000 条 d2 来源行被当作 `(起点, 关系对) → 答案` 的查表记住了（阶段二每行约呈现 2,560 次），而不是由 w2 学到的事实组合出来。佐证：w2 验证整行在 3 个 seed 上是 1.00，但**同一批事实放进训练中从未出现过的单任务格式就几乎全错（0.4%–5%）**——事实知识被绑定在 w2 的 packet 格式里，与第一轮"只在固定宽 2 行里见过的事实无法以单任务形式回忆"的发现一致。
- **D 的诊断读数（在未见关系对上，标注为诊断）**：被监督的读取头在第 1 / 2 轮命中 r1 / r2 的比例 0.59–0.82 / 0.67–0.83，第 3 轮起命中 EOP 1.00——读取规则部分迁移到了新关系对；但实体 probe 在第 1 轮只有 0.14–0.15（x1 = r1(x) 只依赖 `(x, r1)`，而 k = 1 的训练图里 r1 永远只和同一个 r2 同现），第 2 轮起约 0。也就是读得到关系、但 latent 状态里没有可迁移的"第一跳结果"。
- **额外计算**：T 从 8 加到 256，A 的拟合基本不变（s7 掉到 0.95），D 的拟合单调下降到 0.60–0.68（T=32 起开始掉）——受监督的浅题"完成态"在训练预算之外并不稳定；没有任何深度因为加 T 而被解锁。
- **后缀干预**：d3 / d8 改后半段一个关系，答案随之改变的比例 0.79–0.90 / 0.70–0.81，d32 为 0.30–0.51，d128 只有 0.05–0.22；两题同时答对 0。模型在长程序上基本不使用靠后的关系。
- **seed 差异（阶段一）**：seed 123 / 2026 在 20k / 50k updates 学会可重组的 w2；seed 7（A_s7：整行 0.40，第二问 0.40）与 seed 1（B / C / D 的 seed 1 进行中：整行 0.00–0.05）主要在背 w2 行——训练 loss ≈ 0.1 而新配对的验证 loss ≈ 9。A–D 同 seed 共用阶段一，所以这一差异在四个条件里相同。

按方案 §9 的解释表，这是"A–D 都仅 d2（且只有训练对）成功 → 此配方仍无法从浅执行迁移到深执行；不证明所有 latent / loop 架构不可能"的分支，并且比预想更弱：连 d2 未见对都没过，与第四批的 R 同型。配方不改；任何补救（更多关系伙伴 k、单任务格式进训练、降低每行重复次数、正则化等）都应另立实验。
