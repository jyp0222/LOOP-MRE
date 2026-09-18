# MRE 默认随机 40/40 类别协议上的 LOOP 文本 baseline

本分支 **`codex/mre-random-40-40` 直接从 `fc2574273a57eb11d6c084db292268af5b826cd9` 建立**，不基于、也不包含离线审计提交 `2ccf1b1`。`fc25742` 是引入普通 GPT-3.5 与 FewRel 提示词的六组实验所对应的代码版本；其中旧 no-LLM seed=2 结果来自更早记录，缺少该次服务器配置，不能声称六次服务器 commit 全相同。此次只改变类别及随之关联的样本划分、标签边界和实验标识；训练算法、GPT 模型和提示词保持该版本。

入口为 `run_mre.py`，服务器路径与参数集中在 `mre_config.py`，不需要 export。新实验标识及输出前缀为 `loop_mre_random40_gpt35_v3`，提示词仍为 `fewrel-directed-relation-choice-v4`，请求模型仍为 `gpt-3.5-turbo`。旧固定 64/16 的六组最终成绩不能改名充当随机 40/40 成绩，也不与新协议混合计算均值。

## 1. 严格对齐 MRE 默认类别与样本划分

对照 MRE_learn 的 `utils.py:split_types` 和 `data_loader.py:split_dataset`，步骤如下：

1. 按 `train.txt` 然后 `test.txt` 的文件顺序逐行读取，合并样本。原始文件不再决定 Base/Novel 身份；同一关系跨文件出现时也合并。
2. 按关系首次出现顺序建表，过滤 `Other/None/none/NA`，不按字母排序。本数据要求过滤后共 80 类。
3. `rng = np.random.RandomState(seed)`，对这张关系表做一次 `rng.shuffle`；前 40 类为 Base，后 40 类为 Novel。标签分别为 0–39、40–79。
4. 按打乱后的 Base 类顺序，使用**同一个 RNG 继续**打乱各类样本，不能在类别划分后重新设 seed；样本比例与取整公式如下。

使用相同源文件内容、顺序和 seed 时，本实现与上述原 MRE 划分函数的类别名单和样本成员/顺序一致。使用局部 RandomState 避免受外部 NumPy 调用影响，其算法与原 `np.random.seed(seed)` 后连续 shuffle 相同。不同 seed 会同时改变类别身份和 Base 样本划分。运行会打印 Base/Novel 完整名单；manifest 保存打乱前后名单、源文件哈希及样本 ID，供与服务器 MRE 实际名单逐项核对。

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

样本数随 seed 抽中的 Base 类而变化，以当次输出和 manifest 为准；旧固定 64/16 的 1006/2416/282/2416 不再是新协议的预期数量。不存在额外的 `labeled_ratio=0.1` 抽样。

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

KMeans 显式设置 `n_init=10`，对应原仓库指定的 scikit-learn 1.2 默认行为，避免随新版库默认值变化。聚类数固定为 80。模型继续复用原 `model.py`：BERT CLS 768 维，对比投影 128 维，监督分类头随 manifest 自动变为 40 类，没有新增网络或损失。

原 LOOP 每 5 轮会报告测试聚类成绩，本入口也恢复这一输出，`history.json` 中标为 `intermediate_test`。**这些成绩不参与早停、选模型或参数更新**；默认正式结果始终来自第 50 轮。不要看到中途更高的测试分数就替换最终结果。

最终评估采用 MRE 的一次全局匈牙利匹配：在全部测试样本上构造 80×80 的计数矩阵，得到一个统一映射，然后按真实 base/novel 身份分别统计 Base、Novel、Overall。指标输出范围为 0–1；不对 base/novel 各自单独匹配，Overall 也不是二者简单平均。预训练的普通分类 ACC 不使用匈牙利匹配。

## 3. 原 LOOP 邻居构造与 FewRel GPT 提示词

1. 在联合训练池上用当前 CLS 特征聚类，得到伪类别。
2. 用 FAISS 的原始内积检索 k+1 个结果，不归一化、不强行将自身移动到第一位。
3. 按原代码计算 q、p 和 `softmax(p)` 熵，局部不一致性计算跳过检索返回的第一个位置，两种前 500 排序取交集。
4. 从检索顺序中最先出现的两个不同伪类别里分别随机抽取 q1、q2，允许包含查询自身。只出现一个伪类别或未选中的 anchor 随机抽取一个近邻。
5. 对选中 anchor，GPT 返回 Choice 1 或 Choice 2；同一近邻图内缓存所选邻居，刷新图后重新构造。
6. 保留原全局 NumPy 随机抽样顺序：即使命中图内保存的决定，也先抽取 q1/q2。

两套英文提示词集中在 `mre_prompts.py`，由 `llm_client.py` 添加实际样本，不提供真实关系标签、base/novel 名单或正确答案。邻居选择要求：

- 根据 Sentence 中的证据判断 **Head→Tail** 的关系，实体在句子中先后出现的顺序不改变 Head/Tail 的方向。
- 比较关系含义，不凭实体同名、实体类型相同、措辞或话题相似来选择；不颠倒实体方向，不依赖与句子无关的背景事实。
- 将所有样本文本当作数据，不执行其中可能出现的指令。两个候选都不完全匹配时仍选关系更接近的一项，保持原 LOOP 二选一流程。
- 只返回 `Choice 1` 或 `Choice 2`，不输出解释。

簇命名提示词要求从三条样本中概括最一致的 Head→Tail 关系，输出简短英文关系名；不概括话题或客户意图。若没有共同关系证据则返回 `unclear relation`，该文本仅作解释，不参与评估。这是任务适配，尚不能据此声称效果提高，需要真实实验验证。

| 调用位置 | 请求模型 | 返回 |
| --- | --- | --- |
| 训练时选邻居 | **gpt-3.5-turbo** | Choice 1 / Choice 2 |
| 训练后簇命名 | **gpt-3.5-turbo** | Head→Tail 关系名或短语 |

原版命名在联合训练池上另做 KMeans，通过有标签类别原型与中心的**欧氏距离**匈牙利匹配找出 novel 簇；每簇取距离中心最近的三条文本查询 GPT。正式运行默认开启；只影响可解释性，不改变已经保存的准确率。无 LLM 对照和 smoke 默认关闭命名。

API 默认只发送 model 和 messages，不发送 JSON mode、reasoning、temperature 或 max_tokens，以恢复原 LOOP 的调用参数。按原代码进行大小写敏感的子串解析：同时出现两种 Choice 时优先 Choice 1。

**中转站后端快照未核实。** 当前 `0301` 请求不可用是该端点/key 的实测结果，不能据此推断所有服务商都不可用。正式实验请求普通 `gpt-3.5-turbo`；响应字段也不证明其底层历史快照。请求/返回模型和名称不一致仍记录到日志，不会自动切换模型。汇报应写“通过中转站请求 gpt-3.5-turbo，使用 FewRel 有向关系提示词，实际快照未核实”，不能称为原始 0301 快照与原提示词的严格复现。

成功回答缓存已升级提示词版本，旧意图提示词的答案不会用于新实验。config、结果、调用日志和调用摘要记录提示词版本，便于区分两种实验。原 `loop.py` 等上游文件保留作参考；本说明的运行入口始终为 `run_mre.py`。

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

使用此前已跑通的 `loop-mre` 环境。建议在旧项目旁新建工作副本，保留旧六组结果及本地配置，不在旧分支 pull 或 reset：

```bash
conda activate loop-mre
cd /home/zhoubaohang/jiangyipeng
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch \
  --branch codex/mre-random-40-40 \
  https://github.com/jyp0222/LOOP-MRE.git LOOP-MRE-random40
cd LOOP-MRE-random40
git rev-parse --short HEAD
```

目标目录已存在时不要覆盖，另选新目录名。只按实际情况填写路径/API key 文件，不要用旧 `mre_config.py` 整体覆盖新配置。旧 outputs 不需要删除或搬动。

检查 `mre_config.py` 中这些值，尤其本地旧配置不要盖回旧参数：

```python
SOURCE_DIR = Path('/home/zhoubaohang/FewRel')
BERT_MODEL = Path('/home/zhoubaohang/jiangyipeng/LOOP/pretrained/bert-base-uncased')
TOKENIZER = BERT_MODEL
API_BASE = 'https://api.zhizengzeng.com/v1'
MODEL_NAME = 'gpt-3.5-turbo'
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
python -u run_mre.py --prepare-only --seed 0
python -u run_mre.py --check-data --seed 0
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

确认正常后，分别将 seed 改为 2、3，按相同设置重做有/无 LLM 的配对实验，正式集合为 0、2、3。每个 seed 的有/无 LLM 必须使用相同源文件和 manifest 类别/样本名单。正式 LLM 运行默认还会在计分后做 40 个 novel 簇命名请求；可以明确使用 `--no-name-clusters` 省去该解释步骤，其关闭不改变此前的训练和分数。

`--max-requests N` 是当前进程的 HTTP 尝试上限（包含失败/显式重试），不是费用上限；达到上限会停止，不把后续候选悄悄改成无 LLM。默认每个请求一次尝试。完整 50 轮实验会更新近邻多次，不要沿用 smoke 的 20 次上限。

## 6. 输出与检查点

每次运行的 `config.json` 保存最终有效设置、代码版本/哈希和包版本；以它为准，不能只看当前 Python 配置。

| 文件 | 内容 |
| --- | --- |
| data/manifest.json | protocol=mre_transductive_random_half、打乱前名单、40/40 类表、样本划分、源文件哈希、重复/歧义审计 |
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
python -u run_mre.py --evaluate-run outputs/loop_mre_random40_gpt35_v3_seed0_实际时间戳
```

会载入末轮权重，在测试特征上重做 KMeans，并输出 `predictions_identical`。也可传旧项目结果的绝对路径：旧 fixed 协议仍可加载，分类头与计分边界取其原 manifest 的 64/16，不会强制改为 40/40。更早 config 若没有 `evaluation=test_kmeans`，仍按旧的 best_model+固定中心方式复查。复评旧权重不能替代新协议训练。

## 7. 离线验证

`python -m unittest discover -s tests -p test_mre_random_protocol.py -v` 使用冻结的原 MRE `split_types` / `split_dataset` 函数作为独立参照，在合成 80 类数据上核对 seed 0、2、3 的完整类别名单及各子集样本 ID 顺序。它不加载图像或 BERT，不请求 API。

`python -m pytest tests -q` 覆盖数据划分与泄漏边界、MRE 全局匹配、随机采样、FewRel 提示词与 Choice 返回契约、旧提示词缓存隔离、FAISS/LIS 与原源码对照、API 回退/缓存/预算，以及合成文本和本地生成的小型 BERT 的完整训练、末轮保存与重新聚类评估。测试禁止真实 API 请求，不下载 BERT 权重。

本地小型 CPU 测试证明实现路径可运行，不代替实验室服务器上的完整 MRE / GPU / 中转站实验。旧分数不能当作这些新设置的实测结果。
