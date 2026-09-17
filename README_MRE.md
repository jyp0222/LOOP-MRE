# MRE 文本数据上的原 LOOP baseline

本分支按确认的实验要求，恢复原 LOOP 的采样、超参数、模型选择、邻居构造和 GPT 提示词；保留 **MRE 固定 64 个 base / 16 个 novel、MRE 样本协议和 MRE 评估指标**。只使用文本与有方向的实体对，不读取图片。

入口为 `run_mre.py`，服务器路径与参数集中在 `mre_config.py`，不需要 export。当前实验标识为 `loop_original_mre_v1`，输出目录以 `loop_original_mre_seed...` 开头。此前六组采用验证集选 LOOP 模型的结果属于旧设置，不能混入本轮 baseline 的均值。

## 1. 保留的 MRE 协议

类别分别来自原始 `train.txt` 的 64 类和 `test.txt` 的 16 类，按首次出现顺序记录到 manifest；标签编号分别为 0–63 和 64–79，不重新随机选已知类。

对每个有 n 条记录的 base 类：

```python
n_train_val = int(n * 0.5)
n_labeled = int(n_train_val * 0.8)
n_validation = n_train_val - n_labeled
n_test = n - n_train_val
```

约为 **40% 有标签训练、10% 验证、50% 测试**，小类别因取整而不同。所有 novel 样本进入测试池。测试池文本同时作为无标签训练数据；这是传导式（transductive）协议。无标签训练 tensor 中真实标签统一替换为 -1，GPT 不接收真实关系名。

| 数据 | 本次用途 |
| --- | --- |
| base 有标签训练部分 | 预训练 CE、第二阶段 CE 及有标签同类正例 |
| base 验证部分 | **仅预训练**的分类准确率选模型与早停 |
| base 测试部分 + 全部 novel | 无标签训练；中途与最终测试计分 |
| 图片 | 完全不加载 |

当前原始数据预期得到 1006 条有标签训练、2416 条无标签训练、282 条验证和 2416 条测试；联合训练池为 3422 条。运行输出及 manifest 是实际数量的依据。不存在额外的 `labeled_ratio=0.1` 抽样。

`mre_protocol.py` 的原有数据策略保持不变：

- 支持每行 JSON 或 Python 字典，后者用 `ast.literal_eval`。
- 默认实体位置格式为 `indices`：`[13, 14]` 表示两个 token，`[13]` 表示一个。先按原索引插入实体标记，再省略输出中的空白 token，防止实体错位。
- 文本保留 Head → Tail 方向，BERT 与 GPT 均使用这段关系文本；GPT 接收 tokenizer 解码后的实际输入。
- 过滤 `Other`、`None`、`none`、`NA` 关系。重复及冲突记录保留并报告，不自动清洗。
- 保存源文件与生成文件的 SHA-256、样本 ID、类别名单、划分种子和审计信息；每次运行使用独立数据快照。
- 原数据的重复文本可能跨子集；审计中的重复和冲突数量应随实验报告披露。训练不主动使用测试标签。

## 2. 对齐了哪些原 LOOP 设置

这里的“原值”以仓库保留的 `init_parameter.py`、`mtp.py`、`loop.py` 和 `utils/` 原实现为准。

| 项目 | 本次设置 | 原实现位置 | MRE 入口位置 |
| --- | --- | --- | --- |
| 有标签采样 | RandomSampler，普通无放回随机采样；无类别均衡过采样 | dataloader.py | mre_data.py |
| 预训练有标签 batch | **64** | init_parameter.py | mre_config.py |
| 第二阶段 / MLM 联合池 batch | **128** | init_parameter.py、dataloader.py | mre_config.py、mre_data.py |
| 评估 batch | **64** | init_parameter.py | mre_config.py |
| 邻居 k | **50**；FAISS 请求 k+1 个返回值 | init_parameter.py、utils/memory.py | mre_config.py、mre_neighbors.py |
| 预训练 | 最多 100 轮；CE + MLM（mask 概率 0.15） | mtp.py | mre_trainer.py: pretrain |
| 预训练选模型 | 普通分类 ACC（百分数保留两位），严格变好才更新，平分保留较早轮；patience=20 | mtp.py | mre_trainer.py: pretrain |
| 第二阶段 | **50 轮全部训练完成，使用末轮模型，不用验证集早停** | loop.py: train | mre_trainer.py: train |
| 第二阶段损失 | 0.5 × CE + RNCL，温度 0.07 | loop.py、utils/contrastive.py | mre_trainer.py、原 model.py |
| 文本增强 | RTR，替换概率 0.25 | utils/tools.py: view_generator | mre_trainer.py，复用原函数 |
| 优化器 / LR | Transformers AdamW；预训练 5e-5，第二阶段 1e-5 | mtp.py、loop.py | mre_trainer.py: optimizer_for |
| 调度器 | warmup 0.1；总步数 floor(N / batch) × epochs；仍训练尾 batch | mtp.py、loop.py | mre_trainer.py |
| 近邻更新 | 初始及每 5 轮后更新（训练轮次 1、6、11…使用新图） | loop.py | mre_trainer.py |
| 查询筛选 | 熵前 500 与局部不一致性前 500 的交集，不保证 500 次请求 | utils/memory.py | mre_neighbors.py: mine_neighbors |
| 最终预测 | 末轮特征，在**测试集上重新拟合 80 类 KMeans** | loop.py: evaluation | mre_trainer.py: cluster_score_loader |
| 最终指标 | **MRE Base / Novel / Overall** | 本次保留的 MRE 要求 | mre_metrics.py |

KMeans 显式设置 `n_init=10`，对应原仓库指定的 scikit-learn 1.2 默认行为，避免随新版库默认值变化。聚类数固定为 80。模型继续复用原 `model.py`：BERT CLS 768 维，对比投影 128 维，监督分类头 64 类，没有新增网络或损失。

原 LOOP 每 5 轮会报告测试聚类成绩，本入口也恢复这一输出，`history.json` 中标为 `intermediate_test`。**这些成绩不参与早停、选模型或参数更新**；默认正式结果始终来自第 50 轮。不要看到中途更高的测试分数就替换最终结果。

最终评估采用 MRE 的一次全局匈牙利匹配：在全部测试样本上构造 80×80 的计数矩阵，得到一个统一映射，然后按真实 base/novel 身份分别统计 Base、Novel、Overall。指标输出范围为 0–1；不对 base/novel 各自单独匹配，Overall 也不是二者简单平均。预训练的普通分类 ACC 不使用匈牙利匹配。

## 3. 邻居与 GPT 如何恢复

1. 在联合训练池上用当前 CLS 特征聚类，得到伪类别。
2. 用 FAISS 的原始内积检索 k+1 个结果，不归一化、不强行将自身移动到第一位。
3. 按原代码计算 q、p 和 `softmax(p)` 熵，局部不一致性计算跳过检索返回的第一个位置，两种前 500 排序取交集。
4. 从检索顺序中最先出现的两个不同伪类别里分别随机抽取 q1、q2，允许包含查询自身。只出现一个伪类别或未选中的 anchor 随机抽取一个近邻。
5. 对选中 anchor，GPT 返回 Choice 1 或 Choice 2；同一近邻图内缓存所选邻居，刷新图后重新构造。
6. 保留原全局 NumPy 随机抽样顺序：即使命中图内保存的决定，也先抽取 q1/q2。

调用恢复为原提示词，包括原文的 **customer utterance / intent**。虽然输入现在是关系文本，本轮 baseline 按要求不改写成关系抽取专用提示词；这是后续可以单独做消融的地方。

| 调用位置 | 请求模型 | 返回 |
| --- | --- | --- |
| 训练时选邻居 | **gpt-3.5-turbo-0301** | Choice 1 / Choice 2 |
| 训练后簇命名 | **gpt-3.5-turbo**，原 LOOP 的命名调用使用此别名 | 单词或短语 |

原版命名在联合训练池上另做 KMeans，通过有标签类别原型与中心的**欧氏距离**匈牙利匹配找出 novel 簇；每簇取距离中心最近的三条文本查询 GPT。正式运行默认开启；只影响可解释性，不改变已经保存的准确率。无 LLM 对照和 smoke 默认关闭命名。

API 默认只发送 model 和 messages，不发送 JSON mode、reasoning、temperature 或 max_tokens，以恢复原 LOOP 的调用参数。按原代码进行大小写敏感的子串解析：同时出现两种 Choice 时优先 Choice 1。

**0301 后端快照未核实。** 这里恢复的是请求名称；中转站响应里的模型字段也不能证明实际使用了 2023 年的原始权重。请求模型、返回模型及名称不一致会记录到日志，不会偷偷更换请求模型。论文/汇报应写“通过中转站请求 gpt-3.5-turbo-0301，实际快照未核实”。

原代码在请求失败或无法解析时选 q1，本入口恢复该行为，同时发出 warning，记录 `neighbor_fallback`、`fallback_count`，不把失败答案写入持久缓存。已选 q1 仍按原版在本次近邻图内复用。命名失败记录错误。**本地 key 配置错误和请求预算耗尽直接停止**；`--check-llm` 如果只能得到回退答案会报错，不会误报 API 已连通。

## 4. 必须说明的适配边界

本轮对齐的是原 LOOP 的算法与上述设置，并不宣称跨硬件、软件版本和第三方模型逐位复现。

- 数据、类别、样本比例和最终计分改用已确认的 MRE 协议；保留实体方向及标记，max length=128。这是 MRE 输入适配，不是原意图分类数据。
- 修正原监督邻接中的边界问题：仅真正有标签的非负标签参与同类约束，不让第一条无标签样本被当作有标签数据；不执行原近邻挖掘中不影响返回结果的真实标签匈牙利调用。
- GPU FAISS 可用时使用 GPU，测试环境可用 CPU FAISS；训练设备目前为单卡 cuda:0（或 CPU），不自动启用原代码的多卡 DataParallel。
- 极短 batch 没有任何 MLM mask 时令该项为零，避免 NaN；正常 MRE batch 不改变 mask 策略。小于 k+1 的合成测试池会截小 k，正式数据不触发。
- 保留隔离输出、输入审计、有限超时、key 文件、调用日志和持久响应缓存。持久缓存按端点、模型、提示词、候选顺序和生成参数区分；相同请求复用成功答案，可能减少原代码的重复请求。原模型采样参数保持 API 默认值。
- 没有恢复 API key 写进代码等做法；不改变原始 MRE 文件，也不覆盖旧实验。

## 5. 服务器操作

使用此前已跑通的 `loop-mre` 环境，先更新已有工作副本：

```bash
conda activate loop-mre
cd /home/zhoubaohang/jiangyipeng/LOOP-MRE-mre-protocol
git status --short
git -c http.version=HTTP/1.1 pull --rebase --autostash origin codex/mre-fixed-protocol
git rev-parse --short HEAD
```

不要使用 `--ff-only --autostash`，该服务器 Git 不支持此组合。若自动恢复本地配置时出现冲突，先处理冲突再运行；旧 outputs 不需要删除。

检查 `mre_config.py` 中这些值，尤其本地旧配置不要盖回旧参数：

```python
SOURCE_DIR = Path('/home/zhoubaohang/FewRel')
BERT_MODEL = Path('/home/zhoubaohang/jiangyipeng/LOOP/pretrained/bert-base-uncased')
TOKENIZER = BERT_MODEL
API_BASE = 'https://api.zhizengzeng.com/v1'
MODEL_NAME = 'gpt-3.5-turbo-0301'
NAMING_MODEL_NAME = 'gpt-3.5-turbo'
TOPK = 50
LABELED_BATCH_SIZE = 64
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 64
VIEW_STRATEGY = 'rtr'
```

源目录应含原始 train.txt/test.txt，不填以前生成的 TSV 目录。保留已有 Python 3.8、PyTorch 1.11、Transformers 4.15、scikit-learn 1.2 等 LOOP 环境，不需要升级 OpenAI SDK；客户端通过 requests 发请求。恢复 FAISS 检索后可先确认原环境中可导入：

```bash
python -c "import faiss; print('FAISS:', faiss.__version__)"
python -u run_mre.py --check-data
python -u run_mre.py --check-llm
python -u run_mre.py --smoke --seed 0 --max-requests 20
```

`--check-data` 不训练、不调用 API。`--check-llm` 仅一个小型付费查询。smoke 为预训练 2 轮 + LOOP 2 轮，保留正常前 500 排名交集，再限制最多 5 个可查询 anchor，默认不命名；它的数字不作正式结果。若图没有可查询样本会提示覆盖不足，不强制伪造候选。核对 `llm_summary.json` 中的请求次数与 fallback_count。

默认首次实际查询时隐藏输入 key。若希望运行不再等待输入，在仓库之外保存仅含一行 key 的私人文件，设权限为 600，然后填写：

```python
API_KEY_FILE = Path('/home/zhoubaohang/.config/loop-mre/openai_api_key')
```

不要把 key 放入 Git。可先完成 `--check-llm` 再正式训练。恢复原回退策略后，即使训练最终结束，也需要确认失败回退次数，不能仅凭 Finished 断言所有请求成功。

正式 seed 0 对照：

```bash
python -u run_mre.py --seed 0 --no-llm
python -u run_mre.py --seed 0
```

确认正常后，分别将 seed 改为 1、2，按相同设置重做有/无 LLM 的配对实验。正式 LLM 运行默认还会在计分后做 16 个 novel 簇命名请求；可以明确使用 `--no-name-clusters` 省去该解释步骤，其关闭不改变此前的训练和分数。

`--max-requests N` 是当前进程的 HTTP 尝试上限（包含失败/显式重试），不是费用上限；达到上限会停止，不把后续候选悄悄改成无 LLM。默认每个请求一次尝试。完整 50 轮实验会更新近邻多次，不要沿用 smoke 的 20 次上限。

## 6. 输出与检查点

每次运行的 `config.json` 保存最终有效设置、代码版本/哈希和包版本；以它为准，不能只看当前 Python 配置。

| 文件 | 内容 |
| --- | --- |
| data/manifest.json | 64/16 类表、样本划分、源文件哈希、重复/歧义审计 |
| tokenization_audit.json | 截断数量及最大 token 长度 |
| history.json | 预训练分类 ACC、第二阶段 loss、定期间隔测试与调用计数 |
| pretrain_selection.json | 预训练所选轮次、分数、完成轮数；平分保留较早轮 |
| pretrained_backbone/ | 预训练选中模型的 backbone |
| last_model.pt | 第二阶段末轮完整权重，epoch=50（smoke=2） |
| backbone/、tokenizer/ | 可重新载入的末轮 backbone 与分词器 |
| results.json / results.csv | 最终 MRE 指标、实验标识、所用轮次等 |
| predictions.jsonl | 测试样本 ID、真实 label 和预测 cluster ID |
| llm_calls.jsonl / llm_summary.json | API 尝试、缓存命中、回退、返回模型及未核实快照声明 |
| neighbor_queries.jsonl | 候选与所选样本 ID、是否因失败回退 |
| llm_cache.jsonl | 成功回答的持久缓存；不保存 key 或请求原文 |
| cluster_names.json | 训练后命名，不参与指标计算 |

对新结果重新评估（不调用 API）：

```bash
python -u run_mre.py --evaluate-run outputs/loop_original_mre_seed0_实际时间戳
```

会载入末轮权重，在测试特征上重做 KMeans，并输出 `predictions_identical`。旧结果目录的 config 若没有 `evaluation=test_kmeans`，仍按旧的 best_model+固定中心方式复查；不会把旧模型自动解释为新 baseline。

## 7. 离线验证

`python -m pytest tests -q` 覆盖数据划分与泄漏边界、MRE 全局匹配、随机采样、原源码提示词对照、FAISS/LIS 与原源码对照、API 回退/缓存/预算，以及合成文本和本地生成的小型 BERT 的完整训练、末轮保存与重新聚类评估。测试禁止真实 API 请求，不下载 BERT 权重。

本地小型 CPU 测试证明实现路径可运行，不代替实验室服务器上的完整 MRE / GPU / 中转站实验。旧分数不能当作这些新设置的实测结果。
