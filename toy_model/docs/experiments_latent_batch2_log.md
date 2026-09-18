# 第五批落地记录：自主 latent 执行（20 runs）

日期：2026-09-18。方案：[experiments_latent_batch2_20runs.md](experiments_latent_batch2_20runs.md)（不改动）；清单：[manifest](latent_batch2_20runs_manifest.json)。本文记录实现、核查、偏差和作业。

## 1. 作业

| 作业 | 内容 | walltime | 状态 |
|---|---|---|---|
| 3391572 lat2-s1 | run 01 / 05 / 08 / 11 / 15 = A、B、C、D、E 的 seed 1（同 seed 的完整因子组同卡同时跑） | 4 h | 排队 09-18 21:06 |
| 3391573 lat2-s7 | run 02 / 06 / 09 / 12 / 16 = A–E 的 seed 7 | 4 h | 排队 |
| 3391574 lat2-s123 | run 03 / 07 / 10 / 13 / 17 = A–E 的 seed 123 | 4 h | 排队 |
| 3391575 lat2-x | run 04 / 14（A、D 的 seed 2026）+ 18 / 19 / 20（F 的 seed 1 / 7 / 123） | 4 h | 排队 |
| 3391569 lat2-tput | 吞吐 / 评估耗时测试（debug-g 25 min）：10 个进程同卡各训 1,200 updates（`RUN_TAG=_tput`，不影响正式目录），再同时跑一次完整 final 评估计时 | 25 min | 排队 |

20 个 run 每个都是 `train_latent.py --resume`：训练到 250k 后在同一进程内写 `final_eval.json`（幂等）；walltime 不够就用同一条 qsub 命令续跑，不改配方。go39 的 token 余额约 125 node-hours（`show_token`：203,040 中已用 202,914.7），四个 4 h 作业被接受；**之后任何一批都需要新的 token 或改用别的项目额度，这一点需要用户决定。**

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
- **先提交、后拿到吞吐数**：按用户要求 20 个 run 在吞吐测试返回前已提交，打包为每卡 5 个进程；若 4 h 不够靠 `--resume` 续。
- **梯度量级**：只记全局（裁剪前）梯度范数与裁剪触发比例；辅助 loss 分别记"执行轮（t ≤ 2）/ 完成后轮（t ≥ 3）"的 CE，没有逐 loss 单独反传求梯度范数。
- **FLOPs** 为解析估计（线性层 + 注意力矩阵乘，训练 = 3 × 前向），不是实测。
- **唯一布局输入数**未逐个统计（布局空间远大于呈现次数，几乎每次呈现都是新输入）。
- 方案未规定的结构细节：Encoder 末尾一个 LayerNorm；Z 进 decoder cross-attn 前一个 LayerNorm；Encoder 与 decoder 共用输入 embedding，输出头不共享。
- 后缀干预题从 random 类别前 500 题构造，改动位置在 `[d//2, d)`，不引入验证邻接，两题 gold 不同，同一对共用一个布局。
