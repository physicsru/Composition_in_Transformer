# 下一版主方法：对齐 Loop, Think, & Generalize v2

日期：2026-09-19。最新用户决定：模型方法与论文保持一致；同时保留已确认的原子数据联合训练要求，训练语义深度不超过2。

**最新用户决定：主实验做两组比较——D人工按hop设R=d，H模型逐轮学习CONTINUE/STOP。具体见[两组主对照方案](experiments_learned_halting_2026-09-19.md)。本文全序列共享骨干继续适用；随机R、论文固定阈值停止和额外停止标签留作后续候选，不进入当前两组主比较。学习停止属于用户新增扩展，不是论文原样方法；尚未实现或提交。**

状态：Miyabi连接已恢复，已核对服务器world42的meta哈希与原子来源一致。本次交付为方法规范和10k原子来源数据；新模型及训练器尚未实现或接入，没有提交训练作业。w2渲染选择及最终运行manifest仍待确定。本文不是已实现/已运行声明。

## 1. 主模型确定为全序列共享循环

输入一次嵌入后，整个序列的hidden states反复通过同一个GPT-2-style stack：

```text
完整输入 [x, r1, r2, ..., rd]
             |
         Token Embedding
             |
            H0
             |
    同一个4层causal Transformer stack，重复R次
             |
            HR
             |
    Final LayerNorm + 与输入embedding共享的LM head
             |
        一个最终答案实体
```

数学定义：`H_(t+1)=F_theta(H_t; causal_mask, padding_mask)`。

主模型不设置第五批的4个额外latent槽，不使用固定Encoder memory加cross-attention的结构，也不另设答案decoder。隐式思考发生在输入token的隐藏向量上。

| 配置 | 采用值 |
|---|---|
| 共享stack | 4个GPT-2-style decoder blocks |
| 隐藏维度 | 768 |
| Attention heads | 12 |
| Attention | 因果self-attention，加padding key mask |
| 参数共享 | 每轮使用同一套4层参数，不复制独立层 |
| 输入/输出embedding | tied weights，要求同一参数storage |
| 初始化 | attention和FFN残差输出投影置零；不是把整个网络参数置零 |
| 输入注入 | 不在每轮重新加初始embedding |
| 主监督 | 完整程序最终答案的单token CE |
| 额外监督 | 主臂不使用中间实体loss、读取位置loss或显式CoT |
| 内部控制 | 不提供当前关系HEAD、人工指针或中间实体 |

来源：论文Figure1、§3.1–4、AppendixC；作者公开`RecurrentGPT2Block`。这次对齐不是仅把`n_latent=4`改成更大数。

## 2. 两个论文设置分别命名

为保持未来深度外推目标，论文式基线采用§6的设置（最新学习停止主目标见文首增补）：

- **NoPE**，不使用第五批的RoPE和随机稀疏关系槽。
- 训练每batch取`R=clip(Poisson(4), 2, 8)`；完整反传，不将R与题目深度一一对应。
- AdamW，lr=1e-4，weight_decay=0.01，warmup=2000，答案CE的label smoothing=0（与当前公开深度训练器一致）。
- 残差输出投影scale明确设0；作者模型类的默认scale与训练入口传入值不同，必须记录入口最终值。
- 训练从一开始混合原子和d2；不默认继承此前50k w2-only预热。
- **不采用论文加入d3、d4等的课程数据**；这是保留用户目标的明确数据限制，不能说整个实验原封不动复现了§6。

论文§5的两跳参考设置另命名为`paper_systematic_reference`：APE、固定R=4（R1/2/8为其论文对照设置），独立atomic+d2及原子事实留出划分。它可用于阳性复现，但不是本次自动提交的新实验。

NoPE动态方案和APE固定方案不得混在一起称为唯一“论文默认配置”。

### 动态循环次数具体是什么意思

训练程序每个batch独立抽一个R，整个batch共享它。例如第一批R=3，第二批R=6，第三批R=2。同一套4层stack分别被调用3、6、2次，即12、24、8次block计算；参数总量保持不变。每轮继续更新上轮的完整序列hidden states，最后一轮才计算最终答案CE，并通过全部循环反传。

`clip(Poisson(4),2,8)`表示先从参数为4的Poisson分布抽整数，再把小于2的值改为2、大于8的值改为8。这里的4是裁剪前Poisson参数，不应把裁剪后的期望值精确写成4。

模型学习每轮的内部计算；循环次数由抽样程序指定。R不等于语义深度d，一轮也没有被监督为恰好执行一个关系。R=6的d2题仍然只有两跳训练数据，不会因此变成d6。循环增加计算量，不增加输入中的latent token。

目的在于让模型接受多种计算预算，检验是否减少对固定循环次数的依赖；它不保证学会稳定迭代或更深组合。测试先报告所有深度共用的固定R网格，不按真实深度指定“正确”循环次数。

论文§6.3还单独评估输出分布驱动的adaptive halting：相邻轮输出分布KL小于0.01，且当前分布熵小于3.00时停止。这是预设的推理停止规则，不是训练出来的停止头，也不是训练时的随机R。可列为单独推理诊断；使用前需冻结最小/最大预算、分布定义及阈值，报告超预算比例与实际循环数，不用深测试答案调阈值。

## 3. 输入、输出和w2来源

单链查询均使用论文式紧凑格式：

```text
atomic： [x, r]             -> f_r(x)
d2：     [x, r1, r2]        -> f_r2(f_r1(x))
测试dd： [x, r1, ..., rd]   -> f_rd(...f_r1(x))
```

每个实体、关系各占一个token。答案是监督标签，不放入输入；主任务不预测END，不使用TASK/EOP/ANSWER包装，也不生成中间轨迹。

### w2处理的默认建议

已向用户询问w2是否拆为两个独立原子查询，或保留联合双任务扩展。共同的全序列模型决定不依赖此选择。

为尽量贴近论文，默认建议将每个w2来源`((x,r),(z,s))`渲染为两个独立条件项：

```text
[x,r] -> f_r(x)
[z,s] -> f_s(z)
```

两个子题可放在同一batch内。每个子题使用同一模型、相同的完整序列循环。不能把第一题gold答案追加到第二题输入，也不能将d2或深题拆成这种原子调用。

这时w2只保留**来源及曝光**含义，数学上等价于额外原子题曝光，不能宣称仍训练了联合双任务处理能力。原子数据全覆盖时，这些w2子题不会新增独特的单链知识查询。

若用户选择保留双任务共同输入，则必须单列为用户扩展，完整说明跨任务attention、各任务答案位置和loss；不能冒称论文原样协议。用户选择返回后据此更新本节与最终训练manifest。

## 4. 数据规模和损失计数

保留已有world42及来源池：10k atomic、40k w2、10k d2。只有原子和两跳语义进入训练，深题仅评估。

若采用w2独立渲染：来源记录是60k，但去重后的单链输入只有10k atomic +10k d2。w2的80k子题是原子输入的重复，不能再报成额外80k独特训练样本。

建议以论文深度配置的**128个单链查询/update**为训练计数单位：

- 32个独立atomic来源查询；
- 16个w2来源产生32个原子查询；
- 64个d2查询。

对这128个最终答案CE取平均。因此来源比例和loss权重都明确：直接atomic占1/4，w2派生atomic占1/4，d2占1/2。按来源记录计，该batch有112行来源；按模型实际查询计有128题。该建议不是论文原数据配比，只是用户三来源变体的可审计选择。

原来的`mean(答案实体CE)+END CE`、固定256个第五批packet以及50k/200k两阶段配比已不适用于本模型，保存在旧增补中作为历史提议。

总updates、学习率衰减总长度、seed数、训练停止规则与各深度推理预算需在新的运行manifest中一起冻结。当前不把旧250k上限称为已匹配论文grokking预算，也不据此自动申请计算资源。

## 5. 实现时必须明确的论文/公开代码差异

公开代码有两种预测位置：`inp_len`取最后有效输入位置，`last_token`取padding后的张量末位。当前深度训练入口默认`last_token`且右padding；论文语义描述与logit-lens分析则按关系token位置解释。

**本规范按论文语义定义，采用最后有效关系位置读出（`inp_len`），不让padding成为未声明的答案工作槽。** 正式复制发布checkpoint时应读取其真实参数；若它使用padding末位，应单列为代码路径差异，而不是静默改动。

同样须显式保存GPT2Config的activation、各dropout、LayerNorm epsilon及库版本。当前深度入口只显式设置embedding dropout=0，不能推断attention/residual dropout也全为0。论文与systematic脚本在batch/weight_decay等默认值上有差异，按选定协议逐项记录。

## 6. 接入前验证

1. 同一stack的参数在所有循环中共享；tied embedding/output head为同一参数。
2. 在eval模式，给同一`[x,r1]`追加不同r2，r1位置hidden state应保持一致；检验causal mask与位置处理。
3. 目标实体、aux标签、深测试标签不进入输入；训练loss最大语义深度为2。
4. 混合长度padding不改变最后有效位置读出，d128不被截断。
5. 如果w2拆开，检查其批量前向等于两条独立查询的结果；d2/深题必须一次完整输入。
6. 初始残差输出为零、梯度路径有效、最终答案CE确实仅落在规定位置；resume可恢复来源/循环RNG。

保留旧模型和结果作为历史对照；新方法使用独立文件名、配置名和run目录。

## 来源与关联

- [论文v2](https://arxiv.org/abs/2604.07822v2)
- [作者全序列模型](https://github.com/OSU-NLP-Group/Loop-Think-Generalize/blob/main/gpt_utils_extrapolation.py)
- [作者深度训练入口](https://github.com/OSU-NLP-Group/Loop-Think-Generalize/blob/main/train_extrapolation.py)
- [原子来源及旧训练增补](experiments_atomic_joint_next_2026-09-19.md)
- [与旧模型的比较](paper_comparison_loop_think_generalize_2026-09-19.md)
