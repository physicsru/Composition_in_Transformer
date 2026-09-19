# 《Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers》精读摘要

整理日期：2026-09-19。对象：arXiv 2604.07822 **v2**（2026-08-11；v1 为 2026-04-09；页眉 "Published as a conference paper at COLM 2026"），作者 Harsh Kohli、Srinivasan Parthasarathy、Huan Sun、Yuekun Yao（Ohio State University）。

本文读取的来源：

- 论文 HTML 全文 `https://arxiv.org/html/2604.07822v2`（正文 + 附录 A–K）、PDF `https://arxiv.org/pdf/2604.07822v2`（21 页，用来看 HTML 里没渲染出来的 Figure 6 / 19）、v1 HTML（只用来核对附录编号的变化）。
- 作者代码 `https://github.com/OSU-NLP-Group/Loop-Think-Generalize`，main 分支，最新提交 `bb22b19`（2026-09-03）。读了 `README.md`、`getNhopfact.py`、`create_dataset.py`、`train_systematicity.py`、`gpt_utils_systematicity.py`、`train_extrapolation.py`、`gpt_utils_extrapolation.py`、两个 inference 脚本。仓库里**没有**数据文件、日志和 checkpoint（只有 `.gitignore`，实物在 Google Drive），也没有 `requirements.txt`。
- §5 两跳数据的生成器不在上述仓库里：README 写明用的是 `https://github.com/OSU-NLP-Group/GrokkedTransformer` 的流程，即该仓库的 `composition.ipynb`。我读了这个 notebook 的全部代码单元。

标注约定（全文通用）：

- 【论文】= 论文正文 / 附录 / 图的原话或原数；
- 【代码】= 从作者发布的代码里直接读到的；
- 【复算】= 我把作者的数据生成代码原样跑了一遍（只生成数据，没有训练）后统计出来的数；作者真实使用的数据文件不在仓库里，所以复算值只说明"这套生成流程产生什么样的数据"，随机种子不保证相同；
- 【读图】= 从图上目测的近似值，论文正文没给数字；
- 【推断】= 我自己的推论，不是论文或代码的陈述；
- "未说明" = 论文与代码里都没找到。

**附录编号提醒**：v1 的 Appendix F（"Extrapolation with varying model size and dynamic recurrence"，含"只学到 2-hop 时不外推"那句话）在 v2 里是 **Appendix J**；v2 的 Appendix F 是另一个内容（默认初始化的不稳定性）。下文一律用 v2 编号。

---

## 背景

- 【论文 §1】implicit reasoning = 不写 chain-of-thought、在一次前向里把参数里存的知识组合起来。已有工作发现 transformer 即使存了事实，也常常不能在一次前向里做多跳组合。
- 【论文 §1】作者给的机制性解释：普通 transformer 里知识分布在不同层，层间不共享参数；两跳查询要先在浅层取出桥实体、再在深层查第二个事实，如果第二个事实只存在浅层，深层就用不上。训练可以让模型学会组合（Wang et al. 2024a "Grokked Transformers"；Yao et al. 2025），但对"没组合过的事实"和"更深的递归"不泛化。
- 【论文 §2】相关工作：looped / recurrent-depth transformer（Universal Transformer、Saunshi et al. 2025、Geiping et al. 2025 的 Huginn、Zhu et al. 2025 的 Ouro、Fan et al. 2025 的长度泛化）。作者强调与 Fan et al. 的区别：Fan et al. 假设知道每个样本需要的迭代数（oracle），本文不假设。

## 动机

- 【论文 §1】如果同一组层被反复使用，知识就不再绑定在某个深度上，第二轮循环可以复用第一轮用过的"查事实"参数。问题：这样的模型能否在**参数化知识**上做组合泛化。
- 【论文 §3.2】三种泛化的定义（这是全文最重要的定义，下文"与我们设置最相关的细节"第 1 点再展开）：
  - 记原子事实全集 C，随机分成不相交的 C_ID 与 C_OOD。训练集 = **全部**原子事实 C（ID 和 OOD 都以单条事实的形式训练）+ 只由 C_ID 事实组成的 k 跳推断事实的一个子集 I_train。
  - **In-distribution (ID) generalization**：测试 I_k(C_ID) 里没在训练里出现过的推断事实（两跳所用的原子事实都在别的训练组合里用过）。
  - **Systematic generalization**：测试 I_k(C_OOD)，即由"训练里从未参与过任何组合"的原子事实组成的推断事实。原话："induced from atomic facts that are never used in compositions in the training data"。
  - **Depth extrapolation**：测试跳数 k 大于训练最大跳数 k_train 的推断事实（原子事实都是 C_ID）。
- 注意：三个定义里被留出的单位都是**原子事实**或**深度**，没有任何一个定义留出"关系对 (r1, r2)"。

## 方法

**架构**

- 【论文 §4、Figure 1】decoder-only、GPT-2 式 block。L 层的一个 stack 被重复使用 R 次，等效深度 D = L × R：`h^(r+1) = f_θ(h^(r); m)`，r = 0…R−1，m 是"causal attention and padding masks"。循环状态是**整个输入序列所有位置的 hidden state**，没有额外的 thinking token / latent 槽。
- 【论文 Figure 1 caption】"a simple looped transformer similar to Saunshi et al. (2025) without design elements such as input injection, gated halting, and middle looping"——没有每轮重新注入输入、没有门控停止、没有 prelude / coda（整个 stack 都在循环里）。
- 【论文 Appendix C】"For most experiments"：embedding 维 **768**、**12** 个 attention 头、循环块 **4** 层。Appendix J 另做 6 层和 8 层；§5.2 的对照是 8 层普通 transformer；Appendix G 是 8 / 16 / 24 层普通 transformer。
- 【代码】`RecurrentGPT2Block`：`nn.ModuleList([GPT2Block(config) for _ in range(n_layer)])` 直接用 HuggingFace 的 `GPT2Block`，`for _ in range(num_iterations): for block in blocks: ...`；之后 `ln_f`，`lm_head.weight = token_embedding.weight`（输入输出 embedding 绑定，与论文一致）。MLP 宽度、激活用 `GPT2Config` 默认（4×、gelu_new）。
- 参数量：论文未说明。【推断，按代码自行计算】§5 约 30.05M（4 × 7,087,872 + 2207 × 768 + 5 × 768 + 1,536）；§6 约 28.52M（词表 217、无位置表）。

**循环数 R：训练时怎么定、测试时怎么定**

- 【论文 §4】两种策略。fixed：所有训练样本同一个 R。dynamic：每个训练 **batch** 独立采样 `R ~ clip(Poisson(λ), R_min, R_max)`，"Unless otherwise stated" R_min = 2、R_max = 8。λ 在 §4 没写；【论文 Appendix J】最大 8 / 12 / 16 对应均值 4 / 6 / 8；【代码】`--dynamic_mean 4 --dynamic_min 2 --dynamic_max 8`。
- 重要细节【代码】：R 是对整个 batch 设的（`model.num_iterations = ...`），**原子事实行与多跳行用同一个 R**。论文里没有"R = 跳数"这种按样本分配；作者明确反对 oracle 分配（§2、§4）。
- 【论文 §5.1】系统性泛化实验：fixed R ∈ {1, 2, 4, 8}，R = 1 就是 4 层普通 transformer；"We do not use dynamic recurrence in this setting"。测试时的 R：论文未单独说明；【代码】`train_systematicity.py` 评估时 `model.num_iterations = args.recurrence`，即测试 R = 训练 R。
- 【论文 §6.1】深度外推实验：fixed R ∈ {1,…,8} 以及 dynamic。测试时把推理循环数 r 当成自变量扫描（Figure 5：fixed 模型 r = 1…12，dynamic 模型 r = 1…24 外加自适应停止 r*；Appendix I 扫到 40）。【代码】dynamic 模型在课程训练里判断是否过 95% 门槛时用 R = `dynamic_max` = 8。

**位置编码**

- 【论文 §5.1】系统性泛化用 absolute position embeddings (APE)；【代码】`nn.Embedding(n_positions=5, n_embd)`，可学习。
- 【论文 §6.1】深度外推用 NoPE，"which shows better generalization in pilot studies"；【代码】`--positional_embedding_type none`（默认）。

**输入格式与监督**

- 【论文 §3.1、§4】每个实体、每个关系一个专用 token；输入前缀 `<e_h><r_1><r_2>…<r_k>`，目标 `<e_t>`；"We only supervise the next-token distribution at the final position corresponding to the tail entity t"。没有中间实体监督，没有 CoT。原子事实就是 k = 1 的同一格式 `<e_h><r>` → `<e_t>`。
- 【代码】词表 = 实体 + 关系 + 6 个没用到的特殊 token（`<mask> <sep> <a> </a> <q> </q>`），加载时在下标 0 处插入 `<pad>`。标签只取 `target_text` 去掉 `</a>` 后的最后一个 token。
- 【代码】读出位置的实现细节（论文没写，和"最后一个有效 token"不是一回事）：
  - §5：`pad_sequence` 右 padding 到 batch 内最长，loss 取 `logits[:, -1, :]`。训练 batch 是原子行（长 2）与两跳行（长 3）随机混合的，所以原子行实际被渲染成 `<e_h><r><pad>`，答案在 **下标 2 的 `<pad>` 位置**读出——与两跳行的 r2 位置是同一个下标。pad 作为 key 被 mask 掉（−10000），作为 query 仍能看到前面的 token。
  - §6：所有行右 padding 到固定长度 `max_len = 50`，默认 `--pred_pos last_token`，即所有深度的答案都在**下标 49 的 `<pad>` 位置**读出（代码里另有 `inp_len` 选项 = 最后一个有效 token，但不是默认）。
- 【代码】损失：§5 `CrossEntropyLoss(label_smoothing=0.1)`；§6 `label_smoothing=0.0`。论文对 label smoothing 未说明。

**初始化技巧**

- 【论文 §4】把 attention 与 MLP 的输出投影 `c_proj` **零初始化**，使每个循环块在初始化时是恒等映射，理由是共享参数的深网络不稳定（引 Fixup）。§4 是两个实验共用的模型描述，没有把零初始化限定在某一个实验里。
- 【论文 Appendix F】改回默认高斯初始化后，深度外推实验"results are mixed across training runs"：R = 5 的 5 个种子行为差别很大，有的完全不随推理循环数扩展，有的 overthinking 严重；只有 R = 7 表现好。
- 【代码】`gpt_utils_extrapolation.py` 里有这一步（`c_proj.weight *= c_scale`，trainer 默认 `--c_scale 0.0`，bias 置零）；**`gpt_utils_systematicity.py` 里没有**任何零初始化，`c_proj` 是 HF `Conv1D` 默认的 N(0, 0.02)。两个文件里 `nn.Embedding` 都是 PyTorch 默认 N(0, 1)（模型不是 `PreTrainedModel`，HF 的 `_init_weights` 不会被调用）。

**自适应停止（只用于推理，训练里没有停止机制）**

- 【论文 §6.3、Appendix C】Geiping et al. 的规则是相邻两轮输出分布的 KL 小于阈值就停；作者发现这会过早停止，于是加上熵条件：`KL(p_t ‖ p_{t−1}) < ε_KL` **且** `H(p_t) < H_thresh` 才停，ε_KL = 0.01，H_thresh = 3.00。Figure 9 / 10 显示加熵条件后平均迭代数随跳数增长、准确率更好。Figure 5 里的 r* 行就是这个规则。
- 【代码】`forward_adapative` 默认 `eps_kl=0.01, entropy_thresh=3.00`，但 `inference_extrapolation_adaptive.py` 调用的 `evaluate_model_test_adaptive_metrics` 默认 `eps_kl=0.001`（与论文的 0.01 不一致）；最大循环数默认 16。
- 没有可学习的 halting head，没有 ponder cost。

## 实验设置

### A. 两跳系统性泛化（§5）

- 【论文 §5.1】|E| = 2000，|R| = 200，每个实体平均出度 20；40k 原子事实，随机分 95% C_ID / 5% C_OOD；273.6k 条推断事实用于训练；ID 测试集 3k 条（由 C_ID 组成、训练里没出现的两跳），OOD 测试集"nearly 2k"条（由 C_OOD 组成的两跳）。
- 【代码，`GrokkedTransformer/composition.ipynb`】
  - 每个实体随机选 20 个关系（不放回），每个 (h, r) 的尾实体是 `np.random.randint(num_entities)`——**随机尾实体，不是置换**；每个关系只定义在约 10% 的实体上，是多对一的部分函数。
  - 两跳全集 = 对每个 h、每条 (h, r1, b)、每条 (b, r2, t) 枚举，共 2000 × 20 × 20 = 800,000 条。
  - 两跳**只要有一跳是 OOD 事实就不进训练**；两跳都是 OOD 事实 → `test_inferred_ood`；恰好一跳 OOD → 直接丢弃（既不训练也不测试）。
  - 两跳都是 ID 事实的：以 0.005 的概率进 `test_inferred_iid`，其余进训练候选；训练两跳再下采样到 `round(φ × |ID 原子事实|)`，φ = 7.2 → 7.2 × 38,000 = **273,600**。
  - `train.json` = 全部 40,000 原子事实（ID + OOD）+ 273,600 两跳，共 313,600 行；`test.json` 含 `id_atomic / ood_atomic / train_inferred / test_inferred_iid / test_inferred_ood` 五类，每类最多 3,000 条。
- 【代码 README】"In all our experiments, the inferred/atomic ratio is 7.2 with 2000 entities and 200 relations"；数据目录名 `composition.2000.200.7.2`。
- 模型：L = 4，R ∈ {1, 2, 4, 8} 固定，APE；对照：8 层普通 transformer（与 L = 4、R = 2 等 FLOP）。指标：最后位置 argmax 是否等于答案实体。
- 分析手段：logit lens（每层 / 每轮之后过 final LN + LM head，在 r1 位置解码桥实体、在 r2 位置解码答案；Figure 4）；activation patching（Appendix E、Figure 11：破坏 (h, r1) 前缀，再在某个等效深度把 r1 位置的干净激活放回去，看答案恢复多少）。

### B. 深度外推（§6）

- 【论文 §6.1、Appendix B】|E| = 200，|R| = 10，出度 10 = |R|，每个关系是实体集上的一个随机**置换**（每个实体每个关系恰好一条出边）。2k 原子事实；对每个 k ∈ [2, 40] 预先生成 15k 条 k 跳训练事实；每个 k 另有 750 条 held-out 测试。
- 为什么要置换【论文 Appendix B、K】：之前用非置换图时，训练到 40 跳的模型在 80 跳上也"泛化"，但 activation patching 显示因果作用集中在关系序列的末尾几项——跳数多了以后答案几乎由后缀决定，模型学的是"后缀 → 答案"的浅映射。置换关系下答案不能由短后缀决定。
- 课程【论文 §6.1】：先训 原子 + 2 跳，直到 held-out 2 跳测试准确率到 95%；再加入 3 跳，到 3 跳 95% 再加 4 跳……每个阶段都联合训练此前所有数据。某个 k 过不了门槛就终止，永远不会加入 k+1；过了门槛的最大 k 定义为该模型的 **learnable recursion depth**。
- 测试集【论文 §6.1】：每个 k 750 条 held-out；k ≤ learnable depth 的算 ID 测试，k 更大的算外推（OOD）测试，所以划分因模型而异。
- 模型：同 §5，但 NoPE；fixed R ∈ {1,…,8} 与 dynamic（Poisson 均值 4，截断到 [2, 8]）。
- 【代码】课程的实现：每进入一个新阶段都**新建 AdamW 和 scheduler**（优化器状态清零、重新 2000 步 warmup）；门槛判断是每个 epoch 末在 `{k}hop_test` 上 `acc > 0.95`；`--num_epochs 100001` 是累计 epoch 上限；另有默认关闭的开关 `--force_grok`（2 跳阶段强制至少 1000 epoch）、`--use_lr_decay`、`--input_injection`、`--max_grad_norm`（默认 0 = 不裁剪）。

### C. 优化设置

- 【论文 Appendix C】AdamW，lr 1e-4，weight decay **0.01**，linear warmup 2000 步，"in all of our experiments"；batch size：系统性 **512**、外推 **128**。
- 论文未说明：dropout、label smoothing、梯度裁剪、精度、硬件、lr 是否衰减、总步数上限、§5 的种子数（Figure 3 的阴影是"standard deviation"，配的是 100-epoch 滑动平均，没写是跨种子还是窗口内；代码默认 `--seed 42`）。
- 【代码】两个 trainer 都是 `get_linear_schedule_with_warmup`，总步数 = `len(dataloader) × num_epochs`（§5 `num_epochs=150001`，§6 `100001`），实际等于 warmup 后近似常数；bf16 autocast；没有梯度裁剪。dropout：`GPT2Config` 默认 `resid_pdrop = attn_pdrop = embd_pdrop = 0.1`，§5 三者都生效；§6 只把 `embd_pdrop` 设成 `--dropout 0.0`，block 内的 residual / attention dropout 仍是 0.1。

## 实验结果

### 系统性泛化（§5.2）

- 【论文 §5.2】R = 1（4 层普通 transformer）"completely fails"；R = 2 已有 "non-trivial" 的 OOD 准确率；循环多收敛快："R = 4 converges with 2k epoch, while R = 2 takes 7k"，按 wall-clock 也更快（Figure 3 中）。
- 【读图，Figure 3 左，100-epoch 滑动平均】R = 1 全程 ≈ 0.01；R = 4 在约 300 epoch 起飞、约 1k epoch 到 0.7、之后平台 ≈ 0.75–0.80；R = 8 类似，≈ 0.75（曲线到约 3.5k epoch）；R = 2 到约 3.3k epoch 之前一直 ≈ 0，之后缓慢爬到约 0.3（6k epoch），在约 6k epoch 处跳到约 0.7，7–8k epoch 时 ≈ 0.78。**OOD 准确率的平台大约是 0.75–0.8，不是 1.0**；论文正文没有给出最终数值。
- 【读图，Figure 3 中】R = 4 大约 8 小时进入平台，R = 2 约 31 小时才跳升。硬件未说明。
- 【论文 §5.2】三阶段 grokking（R = 4，Figure 3 右）：阶段 1 只有训练准确率上升（记忆）；阶段 2 在记忆之后很久 ID 泛化出现（grokking）；阶段 3 "systematic generalization arises only after the model achieves near-perfect in-distribution accuracy"。正文举例 "10^4 vs. 10^2 epochs"；【读图】图上三条竖线大约在 90、330、1.5k epoch，训练准确率约 10^2 epoch 到 1.0，ID 约 300–400 epoch 到 ≈ 1.0，OOD 从约 300 epoch 开始上升。（正文的 10^4 与图上的阶段 3 标线 ≈ 1.5 × 10^3 不完全一致。）
- 【论文 §5.2、Figure 4】logit lens（L = 4、R = 2 对 8 层普通模型）：阶段 1 不经过桥实体就能给出训练集答案（记忆）；阶段 2 桥实体在 r1 位置可解码，随后 ID 答案在更深的等效深度可解码；阶段 3 OOD 才成功。8 层普通模型只有两个阶段，OOD 上能在很深的层解码出桥实体，但第二跳做不出来（"they lack incentives to encode OOD facts in deeper layers"），"fails to achieve non-zero systematic generalization regardless of training time"。【读图】阶段 3 的循环模型：桥实体从等效深度 1–2 起就在 r1 位置可解码，答案从等效深度 5（第二轮循环的第一层）起在 r2 位置可解码；OOD 的答案解码率明显低于 ID（浅蓝，约 40–75%）。
- 【论文 Appendix E、Figure 11】普通模型在 OOD 上要到第 7 层以后才恢复桥实体，"too late"；循环模型第一轮循环里 r1 位置的所有层都对最终答案有因果作用，后续循环复用同一套参数完成第二跳。

### 深度外推（§6.2–6.3）

- 【论文 Figure 5 面板标题】各模型的 learnable recursion depth：R = 1 → 2 跳；R = 2 → 6；R = 3 → 9；R = 4 → 10；R = 5 → 12；R = 6 → 13；R = 7 → 16；R = 8 → 16；dynamic → 22。
- 【论文 §6.2、Figure 6】dynamic 模型的累计更新步数：学到 4 跳花了 "over 1.3 million steps"，之后到 19 跳只要很少的额外步数；20–22 跳每跳不到 8k 步就能到 > 90%；从 20 跳的 checkpoint 出发只训 21 跳数据，约 50 步就 > 90%。【读图】2 跳 ≈ 0.3M 步、3 跳 ≈ 0.43M 步、4 跳 ≈ 1.33M 步、19 跳 ≈ 1.45M 步、22 跳 ≈ 2.15M 步（batch 128）。
- 【论文 §6.3】用训练时的循环数推理，所有模型都不能超出训练深度；增加推理循环数立刻缓解；"this scaling effect only emerges for R > 4"。课程设置下 R = 6 外推到 17 跳、R = 8 到 24 跳（但两者见过的最大训练深度不同：13 对 16）。
- 【论文 §6.3、Figure 7】统一训练到 12 跳再比较：R = 6 → 14 跳，R = 8 → 19 跳，dynamic → 19 跳。结论：训练时的**最大**循环数决定外推范围；dynamic 的好处是 learnable depth 更大。Appendix H（训练到 8 / 10 / 12 跳的匹配数据）趋势相同。
- 【论文 §6.3、Figure 8】overthinking：logit margin 随推理循环数先升后降；任务越深峰值越低；dynamic 模型衰减慢得多。每轮注入输入（Bansal et al.、Geiping et al. 的做法）"do not resolve the issue in implicit reasoning"。
- 【论文 Appendix I】fixed 模型 2 个种子、dynamic 模型 3 个种子（learnable depth 22 / 21 / 19，Figure 18 标题），趋势一致。
- 【论文 Appendix G、Figure 14】8 / 16 / 24 层普通 transformer：【读图】ID 大约到 4 / 6 / 6 跳；"still do not exhibit OOD generalization to more complex samples"。
- 【论文 Appendix J、Figure 19】generalization ratio = （推理时扩展循环后准确率 ≥ **60%** 的最大跳数）/（训练中已泛化到的最大跳数）。"when models have only generalized to 2-hop composition during training, they do not yet extrapolate to more complex samples"；模型大小（L = 4 / 6 / 8）和最大训练循环数（8 / 12 / 16）对这个比值没有一致的影响。【读图】L = 4、Max R = 8 的曲线在 ID = 2 和 ID = 3 时都是 1.0，ID = 4 时约 1.25，之后在 1.2–1.8 之间波动；dynamic 模型 ID = 22 时约 1.55（约 34 跳）。

## 结论

- 【论文 §7】循环深度 transformer 在从零训练的受控实验里解决了两个组合泛化问题：系统性泛化通过三阶段 grokking 出现；深度外推靠增加推理循环数实现，但受 overthinking 限制。
- 【论文 Appendix A，局限】只研究了一类合成任务；规模小；输入是高度结构化的专用 token，没有自然语言的表面变化、干扰信息；"we do not claim complete and immediate transfer of positive results to LLMs"。
- 【推断】论文证明的"泛化"有三个边界，和我们的目标直接相关：(i) 系统性泛化留出的是原子事实，不是关系对；(ii) OOD 准确率平台约 0.75–0.8；(iii) 深度外推需要训练里已经有 ≥ 4 跳的数据（课程），只训到 2 跳时比值为 1。

---

## 与我们设置最相关的细节

我们的设置（对照用）：500 实体、20 个关系（都是全集置换）、10k 原子事实；两跳训练只覆盖 400 个有序关系对中的 20 个（k = 1 的关系图，每个 r1 只有一个训练后继 r2），这 20 对各自覆盖全部 500 个起点（10k 行）；只监督最终答案；训练对与原子事实都 100%，未见关系对在 chance，输出大多是 r2(x)（`docs/experiments_halting_log.md` §6）。

### 1. 两跳系统性泛化实验到底留出了什么

| 项 | 值 | 出处 |
|---|---|---|
| 实体 / 关系 | 2000 / 200，平均出度 20 | 【论文 §5.1】 |
| 关系的性质 | 每个实体随机拥有 200 个关系中的 20 个；尾实体均匀随机（不是置换，多对一）。每个关系约 200 个头实体、约 190 个不同尾实体 | 【代码】`composition.ipynb` cell 2；【复算】 |
| 原子事实 | 40k，全部进训练集（ID 38,000 + OOD 2,000） | 【论文 §3.2 "all possible atomic facts C"、§5.1】；【代码】cell 5 `all_atomics + train_inferred_facts_ds` |
| 原子事实的划分 | 随机 95% C_ID / 5% C_OOD，按**单条事实**划分，不按实体、不按关系 | 【论文 §5.1】；【代码】`OOD_ratio = 0.05`，`split(atomics, …)` |
| 两跳训练样本 | 273.6k = 7.2 × 38,000，全部由 C_ID 事实组成 | 【论文 §5.1】；【代码】`round(phi * len(id_atomic_facts))`；README "ratio is 7.2" |
| 两跳 / 原子 比 | 7.2（相对 ID 原子事实，作者自己的口径）；相对全部 40k 是 6.84 | 【代码 README】；6.84 为我的算术 |
| 训练覆盖率 | ID–ID 两跳全集约 721.6k，训练用了其中 273.6k ≈ **37.9%** | 【复算】（两个种子：721,638 / 721,753） |
| ID 测试 | 3k 条：两跳都是 ID 事实、但这条 (h, r1, r2) 没进训练 | 【论文 §5.1】；【代码】0.005 概率留出后 `choose(…, 3000)` |
| OOD 测试 | "nearly 2k" 条：**两跳都是 OOD 事实**的两跳 | 【论文 §5.1】；【代码】`if (ent,r1,b) in OOD and (b,r2,t) in OOD`；【复算】1,986 / 2,037 条 |
| 恰好一跳是 OOD 的两跳 | 丢弃，不训练也不测试（约 76k 条） | 【代码】同一段的 `continue`；【复算】 |

**关系对是否被留出？没有。**

- 【论文】§3.2 的定义与 §5.1 的数据描述里没有任何关于留出关系对的内容；论文也没有报告关系对的覆盖情况（未说明）。
- 【复算】把 `composition.ipynb` 的生成逻辑原样跑两次（种子 42 和 7）：训练两跳覆盖了 40,000 个有序关系对中的 **39,954 / 39,948** 个，每个出现过的关系对平均 6.85 条训练样本（中位数 7，最少 1，最多 20）；**OOD 测试的每一条（1,986 / 1,986，2,037 / 2,037）所用的 (r1, r2) 都在训练两跳里出现过**；ID 测试 99.9%（3,558 / 3,562）。
- 【复算】训练里每个 (h, r1) 前缀平均接 **7.20 个不同的 r2**；38,000 条 ID 原子事实每一条都至少参与了一条训练两跳；OOD 原子事实参与训练两跳的条数为 0。
- 所以论文的 "systematic generalization" 的准确含义是：**关系对都见过、每个关系在第一跳和第二跳位置都见过很多次、每个实体也都见过；没见过的是"这两条具体的原子事实被用在组合里"**。这两条事实只以单条 `<e><r>` → `<e>` 的形式训练过。它不是"未见关系对"的泛化。
- 【推断】§5 的世界里，r2 对起点 h 有定义的概率只有约 20 / 200 = 10%，所以"把 r2 直接作用在起点上"这种退路在大多数查询上根本没有对应的事实；我们的世界里每个关系都是全集置换，r2(x) 永远存在。

### 2. 深度外推实验

| 项 | 值 | 出处 |
|---|---|---|
| 世界 | 200 实体、10 关系，每个关系是随机置换；2,000 原子事实，全部进训练（这个实验里没有 OOD 原子事实） | 【论文 §6.1、Appendix B】；【代码】`getNhopfact.py::build_atomic`；【复算】每个关系都是双射 |
| k 跳程序怎么采样 | 均匀抽一条原子事实作起点，之后每步从当前尾实体的 10 条出边里均匀抽一条——等价于在全部 200 × 10^k 个程序上均匀采样；去重，直到 15,750 条；打乱后前 15,000 训练、后 750 测试 | 【论文 Appendix B】；【代码】`build_nhop_facts_fast(max_facts=15750)`、`split_facts(15000, 750)` |
| 训练占全部程序的比例 | 15,000 / (200 × 10^k)：2 跳 **75%**，3 跳 7.5%，4 跳 0.75%，5 跳 0.075%，8 跳 7.5 × 10^−7 | 我的算术；【复算】与生成的文件一致，train / test 重叠为 0 |
| 关系**序列**覆盖 | 2 跳：100 / 100 个关系对全部在训练里；3 跳：1000 / 1000；4 跳：7,806 / 10,000（测试里 587 / 750 条的关系序列在训练里出现过）；5 跳：109 / 750；8 跳及以上：0 / 750 | 【复算】（作者脚本原样运行，种子 42） |
| 训练是否含 > 2 跳 | **含**。课程：原子 + 2 跳 → 2 跳 held-out 到 95% 后加 3 跳 → … 每阶段联合训练之前所有数据；dynamic 模型一路加到 22 跳 | 【论文 §6.1、Figure 5】；【代码】`train_extrapolation.py` 主循环 `count_hops(...) <= k` |
| 测试集 | 每个 k 的 750 条 held-out 程序（同一批实体、同一批原子事实）；k ≤ learnable depth 算 ID，k 更大算外推 | 【论文 §6.1】 |
| 只训 1 跳 + 2 跳会怎样 | "when models have only generalized to 2-hop composition during training, they do not yet extrapolate to more complex samples"；generalization ratio ≈ 1 | 【论文 Appendix J（v1 的 Appendix F）、Figure 19】 |
| 再加 3 跳呢 | 【读图】L = 4 / Max R = 8 在 ID = 3 时比值仍是 1.0，ID = 4 时才到约 1.25；Max R = 12 在 ID = 3 时约 1.33 | 【读图 Figure 19】 |

补充：

- 这个实验里"泛化"有两层：深度 ≤ learnable depth 的 held-out 程序（ID）和更深的程序（外推）。2 跳的 ID 测试是"同一关系对、没见过的起点"——75% 的 (x, r1, r2) 在训练里，全部 100 个关系对都在训练里。**这里同样没有未见关系对的测试。**
- 深度 ≥ 8 的测试程序的完整关系序列从未在训练里出现过（【复算】），所以深层的 ID 泛化确实要求对新的关系序列做组合；但每个相邻关系对 (r_i, r_{i+1}) 都在 2 跳训练里大量出现过。
- 外推的判定门槛不统一：课程门槛是 95%；Figure 19 的比值用 60%；§6.2 里 20–22 跳的"strong generalization"用 > 90%。

### 3. 训练预算

| 项 | 论文 | 发布代码默认值 |
|---|---|---|
| 优化器 / lr / warmup | AdamW，1e-4，linear warmup 2000 步【Appendix C】 | 相同；warmup 后按 `len(dataloader) × num_epochs` 线性衰减，实际近似常数 |
| weight decay | **0.01**，"in all of our experiments"【Appendix C】 | `train_systematicity.py` **0.1**；`train_extrapolation.py` 0.01 |
| batch size | 系统性 **512**，外推 128【Appendix C】 | `train_systematicity.py` **128**；`train_extrapolation.py` 128 |
| label smoothing | 未说明 | 系统性 **0.1**；外推 0.0 |
| dropout | 未说明 | 系统性 0.1（embd / resid / attn，HF 默认）；外推 embd 0、resid / attn 仍是 0.1 |
| c_proj 零初始化 | 有【§4】 | 外推代码有（`c_scale=0.0`）；**系统性代码没有** |
| 梯度裁剪 | 未说明 | 无（外推 `--max_grad_norm 0.0`） |
| 精度 | 未说明 | bf16 autocast |
| 自适应停止 ε_KL | 0.01【Appendix C】 | `forward_adapative` 0.01，但 inference 脚本走的函数默认 0.001 |
| 因果 mask | 有【§4】 | 代码只传 padding mask，因果性依赖 `GPT2Block` 内部自带的 mask。transformers 版本未说明（仓库没有 requirements）。【我在本机 transformers 5.8.0 上核对】该版本的 `GPT2Attention` 已经不在 block 内部做因果 mask（mask 改由 `GPT2Model` 构造），照原样运行会得到**非因果**模型；要复现论文需用旧的 4.x 版本或自己传因果 mask |

README 说 "Most other parameters are set to defaults used to reproduce results in the paper"，所以上面的不一致到底哪边是实际跑出论文图的配置，**未说明**，不能只凭其中一边下结论。

到泛化需要多久：

- 【论文 §5.2】R = 4 约 2k epoch，R = 2 约 7k epoch；R = 1 在图上到 8k epoch 仍 ≈ 0。
- 【推断，换算】1 epoch = `train.json` 全部 313,600 行（【代码】DataLoader 遍历整个训练集）。按论文的 batch 512：613 步 / epoch，R = 4 ≈ 1.23M updates（≈ 6.3 × 10^8 行次），R = 2 ≈ 4.29M updates（≈ 2.2 × 10^9 行次）；按代码默认 batch 128：2,450 步 / epoch，分别 ≈ 4.9M 和 17.2M updates。每条两跳训练行被呈现 2k–7k 次。
- 【读图 Figure 3 左】R = 2 的 OOD 曲线在约 3.3k epoch 之前一直贴着 0（按 batch 512 约 2.0M updates），随后慢升、约 6k epoch 处突跳。也就是说 R = 2 时"很长时间完全没有信号"在论文自己的设置里也是常态。
- 三阶段【论文 §5.2】：记忆（训练准确率先到 1）→ ID 泛化（grokking）→ 系统性泛化（只在 ID 接近完美之后）。【读图】R = 4 的三条标线约在 90 / 330 / 1.5k epoch。
- 循环数的影响【论文 §5.2】：R 越大收敛越快（R = 4：2k，R = 2：7k epoch），且按 wall-clock 也更快；【读图】R = 8 与 R = 4 起飞时间相近，平台略低、波动更大，没有比 R = 4 更好。
- 深度外推【论文 §6.2、Figure 6】：dynamic 模型 2 跳过 95% 约 0.3M updates、4 跳约 1.33M updates（batch 128；前两个数是读图）。【推断，换算】2 跳阶段训练集 17,000 行 = 133 步 / epoch，0.3M updates ≈ 2,250 epoch——即使 75% 的两跳程序都在训练里、100 个关系对全见过，held-out 两跳到 95% 也要这么久。

### 4. 哪些基线失败、差多少

| 基线 | 结果 | 出处 |
|---|---|---|
| 4 层普通 transformer（R = 1），两跳 OOD | "completely fails"；【读图】到 8k epoch ≈ 0.01 | 【论文 §5.2、Figure 3】 |
| 8 层普通 transformer（与 L = 4、R = 2 等 FLOP），两跳 OOD | "fails to achieve non-zero systematic generalization regardless of training time"；只有两个训练阶段；OOD 上桥实体要到第 7 层以后才可解码，答案始终不可解码 | 【论文 §5.2、Figure 4、Appendix E】 |
| 循环模型，两跳 OOD | 【读图】R = 2 / 4 / 8 平台都在约 0.75–0.8，**不是 1.0**；R = 2 需要 7k epoch | 【论文 Figure 3】 |
| 4 层普通 transformer，多跳 | learnable depth 只有 2 跳；无外推 | 【论文 Figure 5 "R=1 (ID 2-hop)"】 |
| 8 / 16 / 24 层普通 transformer，多跳 | 【读图】ID 约到 4 / 6 / 6 跳；弱于同等效深度的循环模型；无 OOD 外推 | 【论文 Appendix G、Figure 14】 |
| 循环数少的循环模型，多跳 | R = 2 / 3 / 4 的 learnable depth 为 6 / 9 / 10；推理时加循环数**没有**外推（"only emerges for R > 4"）；【读图】R = 2–4 在推理循环数超过训练值后连 ID 准确率也下降 | 【论文 §6.3、Figure 5】 |
| 默认初始化（不零初始化） | 各次运行结果不一致，R = 5 的 5 个种子差别很大 | 【论文 Appendix F、Figure 12–13】 |
| 只用 KL 的停止规则 | 过早停止、准确率更低 | 【论文 §6.3、Figure 9–10】 |
| 每轮注入输入 | 不能解决 overthinking | 【论文 §6.3】 |

论文没有给出带数值的结果表；除上面引的原话外，数值都是读图。

### 5. 论文说了什么是泛化的必要条件，以及局限

论文明确说的：

- 权重共享 / 循环是关键："systematic generalization in the 2-hop task already emerges from weight sharing under fixed recurrence"【§5.1】；普通 transformer 无论训多久都不行【§5.2】。
- 需要远超记忆阶段的长时间训练；系统性泛化只在 ID 准确率接近完美之后出现【§5.2】。
- 训练循环数多则快【§5.2】；深度外推要求训练循环数 R > 4，外推范围由训练时的最大循环数决定【§6.3】。
- 多跳需要由易到难的课程（引 Yao et al. 2025："learning k-hop tasks generally requires training the model with an easy-to-hard curriculum"）【§6.1】；见过更深的组合之后才开始外推【Appendix J】。
- 零初始化对稳定性重要【§4、Appendix F】；深度外推用 NoPE 更好（只说"pilot studies"，没给数据）【§6.1】。
- 深度实验必须用置换关系，否则模型学的是后缀捷径【Appendix B、K】。

论文**没有**研究、因此不能引用它来支持的：

- 两跳 / 原子的数据比例：全文只用 7.2 一个值，没有扫描（比例的作用来自它引用的 Wang et al. 2024a，不是本文的结果）。
- 组合的多样性、关系对覆盖率、每个 r1 的后继个数：未说明，也没有消融。
- 未见关系对上的泛化：没有这样的测试。
- 只用 2 跳训练能否到更深：明确的负面结果（Appendix J）。
- 世界规模、模型宽度对两跳系统性泛化的影响：§5 只有 768 维、4 层一个配置。

局限【Appendix A】：单一合成任务族、小规模、专用 token 的结构化输入、不主张直接迁移到 LLM。【§6.3】overthinking 限制了很深的组合。【读图】OOD 平台约 0.75–0.8。

---

## 对本地笔记 `paper_comparison_loop_think_generalize_2026-09-19.md` 的核对

逐项对过论文和代码，**没有发现数值错误**：2000 / 200 / 出度 20、40k、273.6k、95 / 5、3k / 近 2k、6.84 与 7.2、200 / 10 置换世界、15,750 → 15,000 + 750、课程与 95% 门槛、Figure 7 的 d12 → d19、dynamic 到 d22、Appendix J 的 60% 门槛、2k / 7k epoch 及其 1.226M / 4.291M updates 的换算（前提是 batch 512、1 epoch = 313.6k 行，两者我都核实了）、代码 batch 128 / wd 0.1 / label smoothing 0.1、系统性代码里没有零初始化——都对。

需要补充或改得更精确的地方：

1. §3.1 表里"精确关系对覆盖未在PDF列出"是对的，但可以用生成器补上：【复算】39,954 / 40,000 个关系对在训练两跳里出现，OOD 测试 100% 的条目用的是训练里见过的关系对，每个 (h, r1) 前缀平均接 7.2 个不同 r2。这把笔记里"这两种 OOD 不是同一个测试"从定性判断变成了有数的事实。
2. 笔记没写 OOD 测试要求**两跳都是** OOD 事实、恰好一跳 OOD 的两跳被整个丢弃；也没写 §5 的关系是随机尾实体的部分函数（不是置换）。
3. 笔记 §2 表里"最后位置 hidden state"：按发布代码，"最后位置"是右 padding 之后 batch 的最后一列（§5 里原子行是下标 2 的 `<pad>`，§6 里所有行都是下标 49 的 `<pad>`），不是最后一个有效 token。我们第六批的实现读的是"最后一个有效输入位置"（`experiments_halting_log.md` §1），与发布代码的默认不同；论文正文对此未说明。
4. 笔记没提 dropout：HF `GPT2Config` 默认 0.1 在两个 trainer 里都（至少部分）生效，而我们是 0。
5. 笔记没提因果 mask 对 transformers 版本的依赖（见上文第 3 点的表）。
6. 笔记引用的是 Appendix J（v2 编号），正确；任务说明里说的 "Appendix F" 是 v1 的编号。
7. 笔记 §4.4 的 updates 换算只给了 batch 512 的口径；按代码默认 batch 128 是 ≈ 4.9M / 17.2M。哪个是实际配置未说明。
