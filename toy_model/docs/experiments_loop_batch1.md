# 第四批：局部执行器与内部循环（`experiments_loop_next.md` 的可行性审查与落地记录）

日期：2026-09-17。方案：[experiments_loop_next.md](experiments_loop_next.md)。作业：3385016–3385020、3385035（见 §5）。

## 0. 结论先行：方案可行，已按原样实现并提交；两处需要提前说明

1. **方案在现有 pipeline 上完全可落地**：数据复用冻结的 C_k1（k=1，world 42），训练日程、优化器、batch 口径与 `train_chain.py` 一致，只需新增局部协议、按调用重排的位置编码、共享模块的循环前向，以及 d64/d128 的测试题。全部实现（§2），CPU smoke 与故障注入测试通过（§3），六个作业已排队（§5）。smoke 没有引起设计改动，方案原文不变。
2. **L 系列的"成功"几乎是必然的，方案自己也这么定位**：在 L 的接口下模型只需学一张 10,500 项的表（10,000 个 `(x, r)` 更新 + 500 个 `(x, EOS)` 停止），而此前每个实验的原子准确率都是 1.0；表全对时任意深度由 §9.3 的归纳论证保证。所以 L 的信息量在于：多少 updates 收敛、是否**全表**正确（99% 与 100% 在 d=128 上差别巨大：0.99^129 ≈ 0.27）、真实调用与表重放是否一致、以及 **L / L-w2 / L-history 三者的差异**。方案 §0 的"L ≈ L-w2 是合理的正结果，不能据此声称 d2 触发了自主递归算法"我完全同意。
3. **R 有一个方案没有写明的已知事实**：不循环的答案级模型（第三批 chainA，同样的 C_k1 数据、25 万步）在 d2 新关系对上只有 0.003–0.010，即 chance。R 保持"答案级 NTP、不加中间监督"（§6），如果循环不能先解决 d2 新对，d ≥ 3 的读数就没有信息量。方案 §11 已预期"R 比 L 更不确定"，我按原样跑 R；**另外加了一个方案外的补充臂 RC**（同样的循环模型 + 协议 C 的关系/实体 CoT 输出），标为补充，不替代 R 的主读数。

其余细节都可以按方案执行；下面逐节对照。

## 1. 逐条可行性审查（方案章节 → 判断 → 处理）

| 方案 | 判断 | 处理 / 偏差 |
|---|---|---|
| §0 表：F = C_k1 | 已有（3 seed，d2 新对 1.0，d ≥ 3 为 0） | 不重跑 |
| §0 表：O = 外部控制循环 | 可行。注意：C_k1 三个 seed 原来的 `final_eval` **没有** `stepwise_external`（该诊断是 C_k1 跑完之后加进评估的；只有 seed 1 做过 300 题/格的手工核查，方案 §1 也这样说） | 作业 3385021（`MODE=O`）对三个 seed 的 final（250k）和 best_by_val 在扩展测试集上重跑自由 rollout + 外部逐步调用，5000 题/格，d 到 128。smoke（seed 1，3 题/格）：d64 / d128 外部逐步 = 1.0，自由 rollout d ≥ 3 = 0 |
| §2 严格边界 | 可满足：新臂只从 w2/d2 编译调用，不生成 d3+ 任何形式；选点只用 w2 验证 | 训练里没有任何 d ≥ 3 的题、标签或奖励；监测集只写进 metrics，不进入选点 |
| §2.3 披露 | L 是接口改变（每次前向看到的是局部观察，不是完整 w2/d2 序列），R 是完整输入 | 两类分开报告；manifest 记录唯一来源题数、每 update 的调用数、可见 / 监督 token 数 |
| §3 数据 | 复用 C_k1；d64/d128 需要新题 | `data/chain_loop`：`train_w2 / train_d2 / val_d2 / val_w2 / vocab` 五个文件的 hash 与 `data/chain_chainC_k1` 完全相同，`tests.json` 的前 90,000 题与 C_k1 逐题相同，之后追加 d64、d128 各 3 × 5,000 题。random 类别在 d=128 排除验证边的接受率只有 0.95^127 ≈ 0.15%，`builder_chain.py` 把该类别的采样上限按接受率提高（d ≤ 32 的抽样流不变，已核对）。d64/d128 的题全部含重复关系，98–100% 回访实体（20 个关系、500 个实体下必然）；分层字段照旧记录，但"无回访"层在 d=128 为空 |
| §4 L 的观察 / 动作 / 推理规则 | 可行 | `src/data/local_format.py`：`OBS STATE e HEAD r ANS → STEP e' ADVANCE END_CALL`、`OBS STATE e HEAD EOS ANS → HALT e END_CALL`；运行程序 `train_local.run_tasks`：每次调用重置上下文、只用模型自己的实体、无强制解码、每次最多 6 token、调用上限 d+2、记录原始 token / 动作 / 实体 / HEAD / 指针 / 停止原因；EOS 处 STEP = 越界（`overrun`）、关系处 HALT = 提前停止（`early_halt`）、其它 = `format` |
| §5.1 L-w2 | 可行 | `--w2_only`：第二阶段 256 条 w2 呈现替代 128 w2 + 128 d2；每次呈现同样是 2 个 STEP + 1 个 HALT 监督调用；曝光按 `(x, r)`、`(x, EOS)` 逐项累计（`exposure.json`，metrics 里每次评估记录 min/max） |
| §5.2 L-history | 可行；是最有信息量的臂 | `--history`：同一任务（一道 d2，或 w2 的一个子问题）的调用拼成一条序列，**RoPE 位置在每次调用重新从 0 开始**（`gpt2.py` 新增 `pos_ids`），因果 mask 按真实 token 顺序；w2 两个子任务之间重置；训练用正确中间状态，测试用模型自己的。训练序列最多 3 次调用（29 token），测试 d=128 时 129 次调用 ≈ 1,290 token，位置全部在 0–9 内重复——历史是否被忽略正是要测的。`max_len 4096` |
| §6 R | 可行；见 §0.3 的保留 | `train_chain.py --loop_T 2,4,8 --zero_init_out --vocab_ext --select_by w2`：2 层共享模块每个 update 独立抽 T∈{2,4,8}，attention / MLP 输出投影零初始化（模块起始为恒等），位置不变、无循环编号；测试规则 `T(d) = min(128, ≥ max(8,d) 的最小 2 的幂)`（d ≤ 8 → 8，16 → 16，…，128 → 128），另报固定 T=8；深度 × T 的全矩阵（T ∈ {1,…,128}，500 题/格）在 50k / 150k / 250k 的检查点上算（`loop_matrix_*.json`），不用于选点 |
| §7.1 日程 | 与 train_chain 相同：50k w2-only + 200k 混合，batch 256 来源题，AdamW 3e-4 / wd 0.1 / warmup 100，dropout 0，每 5k 存 `ckpt_*.pt` + `last.pt`（优化器、RNG、数据流、HALT 选择 RNG、曝光计数） | L 系列与 R 同规模（d_model 256、2 层、2 头、扩展词表 532 token，1.62M 参数） |
| §7.2 曝光匹配 | 可行 | 每次呈现 2 STEP + 1 HALT 监督；w2 的 HALT 由冻结 RNG（seed 派生，L 与 L-history 相同）每次呈现均匀选一个，另一个可见无损失；`L_row = 0.5·mean(L_STEP1, L_STEP2) + 0.5·L_HALT` 即 STEP token 权重 1/16、选中 HALT token 权重 1/6，除以 256。smoke 核对：40 updates 后 STEP 曝光总和 20,480 = 40×256×2、HALT 监督 10,240、HALT 可见 17,920，全部与算式一致 |
| §7.3 主检查点与选点 | 可行 | 主检查点 250k；240k / 245k 也跑全矩阵（`final_eval_ckpt_24{0,5}000.json`）；次要 best 由 w2-only 验证（20 条 val_w2 = 40 个一步任务的自由执行完整正确率，平手用它们的 teacher-forced loss）→ `best_by_valw2.pt`；L-w2 不碰 d2 标签 |
| §8 测试矩阵与指标 | 可行 | 8 个深度 × 3 类别 × 5,000 题；监测集每格前 500 题；指标：合法停止下的最终答案、完整轨迹、own-update 及其分母、首错位置 / 类型、halted 比例、平均调用数；R 报最终答案 + 计算次数 |
| §9.1 局部表与重放 | 可行 | 每次评估都导出 10,500 项表（动作、实体、合法性、原始 token、logit margin），最终评估写 `table_<tag>.json`；L / L-w2 的监测集与最终矩阵**同时**用真实调用和表重放，并逐题比对动作 / 实体序列（`agreement_real_vs_table`），覆盖每次评估的 500 题/格与最终的 5,000 题/格（超过方案的 100 题/格）。L-history 只有真实调用（表对它不定义） |
| §9.2 四种重放 | 可行 | `replay(update, control)`：缺实体分别记 `missing_update` / `missing_halt_entity` / `format`，不填正确实体；`correct_correct` 在 smoke 中全 1.0（检查运行程序与计分） |
| §10 涌现分析 | 结果出来后做 | `t_d` 读数与热图用汇总脚本从 metrics.jsonl 算，不参与选点 |
| §12 P0 | 完成 | 见 §3 |
| §12 P1 | 已提交 | 12 条方案内轨迹 + 3 条 RC |
| §12 P2 | 未做 | 新世界 43/44、E/R scaling、H 留出版本等冻结配方后再做 |

## 2. 实现

- `src/model/gpt2.py`：`forward(input_ids, pad_mask, pos_ids=None, n_loops=1)`；`pos_ids` 给出显式 RoPE 位置（L-history 每次调用从 0 重排），`n_loops` 让同一组 block 重复应用；`zero_init_out=True` 把每个 block 的 `attn.out` 与 MLP 第二层权重置零。
- `src/data/builder_chain.py`：random 类别的采样上限按接受率提高（d ≤ 32 不变）。
- `src/data/local_format.py`：协议 L 的词表扩展（原词表 + `OBS STATE HEAD EOS STEP ADVANCE HALT END_CALL`）、调用渲染、来源题编译（d2 → 3 次调用；w2 → 2 个独立任务各 2 次调用）、输出解析（只有两种合法形状）、`score_run`（首错类型：format / early_halt / overrun / state_update / halt_entity / no_halt / propagated）。
- `src/train_local.py`：L / L-w2 / L-history 的训练与评估（§1 表）；`--eval_only ckpt --eval_tag`；`--resume` 精确续跑（已测）。
- `src/train_chain.py`：`--loop_T / --loop_cap / --loop_fixed / --loop_matrix_at / --loop_matrix_T / --zero_init_out / --vocab_ext / --select_by w2 / --final_depths / --final_extra_ckpts / --eval_tag`；`stepwise_external` 在循环模型上用 T(1)=8。
- `scripts/test_local_runner.py`：§12 P0.3 的故障注入测试（正确表 → 全对；注入错实体 / 提前 HALT / EOS 处 STEP / 非法 token / HALT 返回别的实体 → 各自的错误类型；表重放与运行程序一致；四种重放隔离更新 / 控制误差；history 运行程序同样通过）。
- `scripts/pbs_loop.sh` + `configs/cells_loop.txt`（`name | script | train args`，共享 `data/chain_loop`；`MODE=O` 跑 C_k1 的重评估）。

## 3. P0 核查记录

1. 复查历史问题：`train_chain.py` 的 decorator 位置、D 的缺失评估都已在第三批修复（commit 047be29）。
2. smoke（CPU，E=60 / R=10 的小世界，20+20 或 20+40 updates，5 题/格）：L / L-w2 / L-history / R / RC 全部跑通（训练、评估、检查点、最终矩阵、240k/245k 等价的额外检查点、best）；L 的 `--resume` 从 40 续到 60；R 的 `loop_matrix_000020.json`、`fixed_T8`、`best_by_valw2.pt` 都生成。
3. 故障注入：`scripts/test_local_runner.py` 全部断言通过。
4. O：seed 1 的 250k 检查点在扩展测试集上（3 题/格）d64 / d128 外部逐步 = 1.0；全量由作业 3385021 完成。

## 4. 事前预测（沿用方案 §11，不改）

L 最可能稳定成功；L-w2 ≈ L；L-history 不确定；R 最不确定（且先看 d2 新对是否 > chance）；RC 若 d ≥ 3 仍为 0，说明循环本身不改变 CoT 模型的按位置规则。

## 5. 作业表

| 作业 | 内容 | walltime | 状态 |
|---|---|---|---|
| 3385016 loop-L | L × seed 1/7/123 | 8 h | 排队 09-17 |
| 3385017 loop-Lw2 | L-w2 × 3 | 8 h | 排队 |
| 3385018 loop-Lhist | L-history × 3（d=128 评估无 KV cache，最慢） | 12 h | 排队 |
| 3385019 loop-R | R × 3 | 8 h | 排队 |
| 3385020 loop-RC | RC × 3（方案外补充；最终矩阵限 d ≤ 32） | 12 h | 排队 |
| 3385478 loop-early | L_early / Lhist_early × 3：主臂前 5k updates 的 200-update 分辨率重跑（§10 的 t_d；不是新臂） | 2 h | 排队 09-18 00:30 |
| 3385035 loop-O | C_k1 三个 seed 的 final / best 在扩展测试集上的重评估（首次提交 3385021 六个评估共用一卡、rollout batch 1000，在 d=128 的自由 rollout 处 CUDA OOM；已改为 `--rollout_batch 250` 重提） | 3 h | 完成 09-17 23:00（§6） |

结果读法：`final_eval_final.json`（主）、`final_eval_ckpt_240000/245000.json`、`final_eval_best.json`；L 系列另有 `table_*.json`、`predictions_*.jsonl`（每次调用的原始 token）、`exposure.json`；R 另有 `loop_matrix_*.json`。

## 6. 结果

### 6.1 O：外部控制循环（作业 3385035，完成 09-17 23:00）

C_k1 的三个 seed，final（250k）与 best_by_val 两个检查点，扩展测试集全量（8 个深度 × 3 类别 × 5,000 题，共 120,000 题/检查点）：

| 读数 | all_seen / one_new / random，d = 2 … 128 |
|---|---|
| 外部逐步调用（程序每步喂 `Q s_t r_(t+1) ANS`，串接模型自己的实体） | **1.0000 在全部 24 格、3 seed、两个检查点**（144 格 × 5,000 题无一错） |
| 原子单步（10,000 个 `(x, r)`） | 1.0000（全部） |
| 自由 rollout 完整轨迹 | d2 = 1.000（三类别），**d ≥ 3 全部 0.000，含 d64 / d128** |

解读：C 的规范单步接口在训练覆盖域上全表正确，因此外部循环到 d=128 也不出错——这正是方案表里 O 的"外部控制正对照"；它同时把第三批"d ≥ 3 为 0"的结论延伸到 d=64 / 128。O 不训练任何新东西，它的 1.0 说明瓶颈只在"谁来选下一个关系、谁来决定停止"，而不在更新本身。L 系列要回答的是：把这两件事也交给模型学（STEP/HALT 动作），但每次调用仍是局部观察时，模型能否学准并在重复调用下保持不变。每个评估耗时 1,640 s（六个进程共用一卡，rollout batch 250）。
