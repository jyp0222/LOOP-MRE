# 在 LOOP 上运行 MRE 文本实验：固定 64 个 base / 16 个 novel

本分支把 **LOOP 的训练方法**接入已确认的 **MRE 样本划分和评估协议**：只使用句子及有方向的实体对，不读取图片；类别固定为原始 `train.txt` 的 64 类和原始 `test.txt` 的 16 类。入口是 `run_mre.py`，服务器路径和默认参数写在 `mre_config.py`。

这是“LOOP 方法在 MRE 协议下的文本实验”，不是复现 MRE 的多模态模型，也不是沿用先前 `mre_64_16/train.tsv` 的实验。原始 LOOP 的说明和入口仍保留在 [README.md](README.md)。

## 1. 本次采用的实验协议

### 类别与样本划分

类别名单从两个原始文件分别提取，按各文件中关系首次出现的顺序固定到 `manifest.json`；base 的全局标签为 `0..63`，novel 为 `64..79`。代码检查类别数及两组类别不相交，不再随机选择已知类。

对每一个含有 `n` 条记录的 base 类，按下式划分：

```python
n_train_val = int(n * 0.5)
n_labeled = int(n_train_val * 0.8)
n_validation = n_train_val - n_labeled
n_test = n - n_train_val
```

因此比例约为 **40% 有标签训练、10% 验证、50% 测试**；小类别的实际比例由整数取整决定，不能直接按总样本数乘比例。每个 base 类的三个子集必须非空，不满足时会报错。所有 novel 记录进入测试池。

| 数据部分 | 用途 | 训练时能否读取真实标签 |
| --- | --- | --- |
| base 的有标签训练部分 | CE 监督及联合训练 | 能 |
| base 的验证部分 | 选择 checkpoint、早停 | 不进入训练 loss，仅用于验证 |
| base 的测试部分 + 全部 novel | 无标签训练，同时用于最后测试 | 无标签训练 tensor 中统一为 `-1` |

**这是一种传导式（transductive）实验**：待评估的测试文本在训练时可见，但测试真实标签不能用于训练、选邻居、问 GPT 或选 checkpoint。这正是本次确认的 MRE 样本协议。测试与无标签文件的样本 ID、文本及顺序必须相同；验证样本 ID 必须与训练/测试分离。

不存在额外的 `labeled_ratio=0.1` 抽样。划出来的 base 有标签训练记录全部可用于监督。此前使用 211 条有标签记录、481 条测试记录等划分得到的数字，不能直接与本分支的新结果比较。

### 数据处理与审计

- 原始文件支持每行 JSON 或 Python 字典文本；使用安全的 `ast.literal_eval` 解析后者，不使用 `eval`。
- 输入可以使用 `tokens` 或 `token`；若两者同时存在但内容不同则报错。实体 `h`、`t` 的方向保留为 `Head → Tail`。
- 空字符串和仅含空白的 token 可正常处理：先按原始 token 索引定位实体、插入标记，再在输出句子中省略空白内容，不删除原始记录、不改变实体索引。manifest 审计记录空白 token 的数量和行号/位置示例。非字符串 token、全空白句子或全空白实体范围仍会报具体位置，避免把损坏输入悄悄当作正常文本。
- 默认 `POSITION_FORMAT = 'indices'`，例如 `[13, 14]` 表示两个连续 token，`[13]` 表示一个 token。只有数据确实采用半开区间时才改为 `'half_open'`；代码不会根据数组长度猜测。
- 仅过滤关系名 `Other`、`None`、`none`、`NA`；不会加载或检查图片文件。
- **重复记录和同输入多标签记录保留并报告，不自动去重、不自动删除歧义数据。** 不同原始行可能具有同样文本，因此仍可能出现跨子集的文本重复；相关计数记录在 manifest 的 `audit` 中，汇报实验时应披露。
- 保存原文件 SHA-256、样本 ID、类别顺序、种子、各类计数以及生成文件的 SHA-256。加载时检查文件与清单一致。

每次运行都创建独立输出目录及独立的数据快照，不覆盖之前的实验。

## 2. 评估究竟改了什么

`mre_metrics.py` 对预测 cluster ID 与真实类别构建一个 `80 × 80` 计数矩阵，**只进行一次全局匈牙利匹配**，再基于该映射计算：

- `Base`：真实类别属于 base 的样本准确率。
- `Novel`：真实类别属于 novel 的样本准确率。
- `Overall`：所有测试样本的准确率。

三项均为按样本计数的准确率，输出范围 `0..1`，不提前四舍五入；例如 `0.46` 对应 `46%`。不能给 base 和 novel 分别做一次匹配，也不能把 Overall 直接算为 `(Base + Novel) / 2`。验证集没有 novel，故验证输出的 `Novel` 是 `null`。

**MRE 和原 LOOP 的这些准确率具有相同的基本全局匹配思想。** 本次不能声称“换指标公式就能提高分数”。实际需要统一的是数据、类别、推理和 checkpoint 选择口径：

1. 预训练和 LOOP 训练都以 base 验证准确率选择模型；相同分数保留较后的 checkpoint，连续 `PATIENCE` 轮未达到当前最佳则早停。
2. LOOP 阶段仅在训练池特征（有标签训练 + 无标签池）上拟合 80 个聚类中心。每轮验证前，用当前权重重新拟合训练池中心，并将最佳模型权重与其中心一起保存。
3. 验证和最终测试都只做“提取特征 → 最近中心预测 → MRE 匹配计分”，**不会在 `score_loader` 中重新拟合验证集或测试集的 KMeans**。由于实验本身是传导式，训练池原本包含测试文本；这不等于测试时另行重聚类。
4. 训练结束恢复验证选出的权重和中心，只进行最后一次测试汇报，不用中途测试成绩选最佳轮数。

这里沿用的是 MRE 的评估和选择协议；LOOP 的预测器仍是特征聚类中心，并没有被替换为 MRE 模型的参数化分类头。匈牙利映射只用于计分，不反馈到训练。

## 3. 保留的 LOOP 方法与必要调整

主干继续使用原 `model.py` 的 BERT 和投影头、`utils/contrastive.py` 的对比损失：BERT CLS 特征为 768 维，投影后的对比特征为 128 维，监督分类覆盖 64 个 base 类，聚类覆盖全部 80 类。

```text
有标签 base + 无标签文本
  → 预训练：CE + MLM
  → CLS 特征、训练池聚类、近邻图
  → 按熵与邻域不一致性筛选候选样本
  → GPT 比较有方向的实体关系，选择较相近的候选
  → LOOP 训练：0.5 × CE + RNCL
  → base 验证选择“权重 + 中心”
  → 固定中心预测，输出 MRE Base / Novel / Overall
```

GPT 只接收 `Head / Tail / Sentence` 和候选文本，不接收真实关系名或标签。返回严格 JSON 的零起始候选索引；API 报错、拒答、输出不完整或索引非法都会停止，不随机补答案或偷偷换模型。

为使新协议可运行且可审计，以下实现差异需要在报告中说明：

| 部分 | 本分支的处理 |
| --- | --- |
| 有标签采样 | 按 MRE 使用类别均衡采样：打乱类别顺序循环，从当前类内有放回抽样 |
| 输入视图 | 沿用之前文本实验的无额外增强设置；不加入图片、实体视觉特征或新损失 |
| 优化器 | 使用 PyTorch AdamW，`eps=1e-6` 对齐旧 Transformers AdamW 的默认 epsilon；不承诺跨库逐位相同 |
| 更新步数 | 使用实际 batch 数，包含最后一个不满 batch |
| 近邻 | 保留原始内积相似度；分块 NumPy 实现免除新入口对 GPU FAISS 的强制依赖，明确把自身放在近邻列表首位 |
| GPT 候选 | 排除查询本身，候选来自不同伪类别；未查询样本仍沿用随机近邻行为 |
| LIS 计算 | 保留原代码实际的 `q ** 2 / 2`、归一化及 `softmax(p)` 熵排序；本次没有悄悄改为另一套论文公式 |
| 标签边界 | 只有非负真实训练标签能构成监督同类对；所有无标签样本的 `-1` 不会被当成“同一个类” |
| 文本长度 | BERT 默认截断到 128 token；GPT 接收完整关系文本，二者可见长度不同，截断统计写入 `tokenization_audit.json` |

默认超参数来自配置文件：预训练最多 100 轮、LOOP 最多 50 轮、patience 10；batch 为有标签 16、LOOP 24、评估 32；学习率分别 `5e-5` 和 `1e-5`；top-k 为 20，每 5 轮更新近邻；候选池上限为 500。候选池是两种排序的交集，不保证每次发出 500 次请求。这些默认值不代表已经在 MRE 上调优。

## 4. GPT 模型与 API key

默认使用 LOOP 论文中的 **GPT-3.5 Turbo 系列**，不再选择 GPT-5.6 或 GPT-6。论文第 4.1.4 节写明使用 GPT-3.5 Turbo API；原代码的邻居查询使用 `gpt-3.5-turbo-0301`，聚类命名使用 `gpt-3.5-turbo`。

必须区分模型系列和历史快照：`gpt-3.5-turbo-0301` 已于 2024-09-13 停止服务，见 [OpenAI 停用说明](https://developers.openai.com/api/docs/deprecations)。本项目明确配置官方仍列出的 `gpt-3.5-turbo`，见 [GPT-3.5 Turbo 模型文档](https://developers.openai.com/api/docs/models/gpt-3.5-turbo)。这是同系列复现，不能声称使用了当年的相同快照；客户端拒绝 `0301` 配置，不会自动替换用户指定的模型。账号权限与网络连通性仍由实际请求确认。

```python
# mre_config.py
MODEL_NAME = 'gpt-3.5-turbo'
REASONING_EFFORT = None  # 不向 GPT-3.5 发送推理参数
```

GPT-3.5 没有本项目此前 GPT-5/6 的推理档位；`None` 表示该参数不适用。旧命令的 `--reasoning-effort none` 仅兼容解析，不会发送到 API。可显式指定客户端支持的 GPT-3.5 快照，实际可访问性以账号请求为准。`llm_calls.jsonl` 同时记录请求模型和 API 返回的实际模型版本；非 GPT-3.5 响应会被拒绝。

客户端通过 `requests` 请求 OpenAI `/v1/chat/completions`，使用 JSON mode（`response_format={"type":"json_object"}`），再在本地校验答案字段与候选编号；不发送 GPT-3.5 不支持的严格 JSON Schema 或推理参数。不依赖新版 OpenAI SDK，无须为此升级服务器已有的 `openai==0.28.0`。配置 `MAX_OUTPUT_TOKENS=4096` 转换为该接口的 `max_tokens`，限制生成答案的 token 数；截断、空回答、拒答和非法 JSON 都不会作为训练监督使用。相对原版 `Choice 1/2` 文本，这里保留当前 MRE 分支的 JSON 答案契约，提示词仍比较有方向的关系。

缓存版本和请求端点已经更新，旧 GPT-5/6 回答不会混用。客户端不显式设置采样 temperature，与原 LOOP 调用同样采用 API 默认值；重复请求仍不保证完全确定。模型、token 上限、提示词、候选顺序或文本变化会形成不同缓存键。

不需要 `export`。默认首次实际请求时隐藏输入 API key。需要非交互运行时，把 key 保存为服务器上仓库之外的私人文本文件，并在配置中指定：

```python
API_KEY_FILE = Path('/home/zhoubaohang/.config/loop-mre/openai_api_key')
```

该文件仅放一行 key，不要把 key 写进 Python 代码、Git 或实验记录。通过终端编辑文件时可将权限设为 `chmod 600`。显式指定的 key 文件不存在或为空会报错，不会自动换用其他身份。客户端也兼容 `OPENAI_API_KEY`，但这不是必需配置。

API 请求会按账号实际用量计费。`--max-requests N` 限制的是本次进程的 **HTTP 尝试次数，包含重试**，不是费用上限；缓存命中不消耗该次数。默认只对网络错误、429 或 5xx 做有限重试，401/403 等不会连续重试。缓存与日志不保存 API key 或请求原文，查询记录通过样本 ID 回溯。

## 5. 在服务器上逐步运行

以下命令在服务器已有的 `loop-mre` 环境执行。代码使用 Python 3.8 兼容语法，可沿用原 LOOP 的 torch/Transformers/scikit-learn/scipy/numpy/pandas/tqdm 环境；新 API 客户端还需要 `requests`。不要为了运行新入口直接升级整个旧环境。BERT 权重及 tokenizer 继续使用服务器已验证成功的本地目录。

**第一步：取新分支到独立目录。** 下面使用新的 `LOOP-MRE-mre-protocol` 目录，保留原来的 LOOP 目录和实验输出。

```bash
conda activate loop-mre
cd /home/zhoubaohang/jiangyipeng
git clone --branch codex/mre-fixed-protocol --single-branch https://github.com/jyp0222/LOOP-MRE.git LOOP-MRE-mre-protocol
cd LOOP-MRE-mre-protocol
```

如果该目录已经存在，请进入对应工作副本后检查分支和本地修改，不要再次覆盖克隆。

**第二步：编辑 `mre_config.py`。** 至少确认：

```python
SOURCE_DIR = Path('/home/zhoubaohang/FewRel')
BERT_MODEL = Path('/home/zhoubaohang/jiangyipeng/LOOP/pretrained/bert-base-uncased')
TOKENIZER = BERT_MODEL
MODEL_NAME = 'gpt-3.5-turbo'
REASONING_EFFORT = None
```

`SOURCE_DIR` 下应是未经此前转换/重划分的 `train.txt` 和 `test.txt`。不要填入以前的 `data/mre_64_16` TSV 目录。路径也支持通过 `--source-dir`、`--bert-model`、`--tokenizer` 传入；如果改变模型路径而分词器也随之变化，要同步设置分词器路径。

**第三步：先验证数据，不调用 API。**

```bash
python run_mre.py --prepare-only
python run_mre.py --check-data
```

前一条仅解析、划分、审计；后一条还加载本地 tokenizer 并检查 tensor。每条命令都产生自己的输出目录。重点检查固定类别 `64 / 16`、各子集数量、重复/歧义计数和截断比例；不要期待得到旧版 211/2734/262/481 的数量。

**第四步：独立检查 GPT。此命令会发起一个小型付费查询，不训练 BERT。**

```bash
python run_mre.py --check-llm
```

应看到请求模型、API 返回的实际模型版本和 `Choice 1` 或 `Choice 2`。如果返回权限或模型错误，先修正 key/账号权限/配置，不要绕过错误继续训练。之前服务器到 `api.openai.com` 的连接超时不会因更换模型自动解决。

**第五步：短流程检查。**

```bash
python run_mre.py --smoke
```

默认将预训练与 LOOP 训练各设为 2 轮，并把每次近邻更新的待查询 anchor 上限降为 5；默认不生成簇名称。在默认更新频率下，短流程只更新一次近邻。实际查询可能少于 5 个甚至为 0，不能仅凭“训练结束”断言训练中一定用了 GPT，应检查 `llm_summary.json` 和 `neighbor_queries.jsonl`。HTTP 重试可能使尝试数大于 anchor 数；必要时另加 `--max-requests`。短流程指标只用于确认流程可运行。

**第六步：正式实验与无 LLM 对照。**

```bash
python run_mre.py --seed 0
python run_mre.py --seed 1
python run_mre.py --seed 2

python run_mre.py --no-llm --seed 0
python run_mre.py --no-llm --seed 1
python run_mre.py --no-llm --seed 2
```

`--no-llm` 在同一套 MRE 数据、训练和评估流程下禁用 GPT，保留随机近邻选择，是判断 GPT 是否带来收益的必要对照。GPT-3.5 的有限请求短测试示例：

```bash
python run_mre.py --model gpt-3.5-turbo --smoke --max-requests 20 --seed 0
```

此处 `seed` 同时影响样本划分与训练随机性；换 seed 不是仅换模型初始化。应报告三个 seed 各自结果以及均值、标准差，并明确这是跨划分和训练随机性的统计。同 seed 的方法比较还应核对原文件哈希、manifest、参数和库版本；不要仅凭种子数字相同就声称所有随机过程完全配对。

## 6. 输出、复核与后续汇报

每次运行输出到 `outputs/mre_seed<seed>_<时间>/`。不同执行模式只生成相应阶段的文件；完成训练后主要文件如下：

| 文件 | 用途 |
| --- | --- |
| `config.json` | 实验参数、Python/依赖版本、Git commit、本地修改标志、源码与 manifest 哈希 |
| `data/manifest.json` 及四个 JSONL | 类别、样本划分、数据审计、测试标签与无标签训练文件的明确分离 |
| `tokenization_audit.json` | BERT 截断数量和原始 token 长度统计 |
| `history.json` | 预训练/LOOP loss、base 验证成绩及调用计数 |
| `best_model.pt` | 最佳权重及与其对应的 80 个训练池中心 |
| `pretrained_backbone/`、`backbone/`、`tokenizer/` | 预训练主干、最终选中主干和分词器 |
| `results.json`、`results.csv` | 最终 `Base / Novel / Overall`、最佳轮数和实验信息 |
| `predictions.jsonl` | 每个测试样本 ID、真实标签和预测 cluster ID |
| `llm_calls.jsonl`、`llm_cache.jsonl` | 请求状态、模型元数据、token 用量及严格校验后的缓存答案 |
| `neighbor_queries.jsonl`、`llm_summary.json` | 实际询问的样本/候选 ID，以及本次 HTTP 尝试与缓存命中计数 |

没有实际 GPT 查询时，相应逐条日志文件可能不存在。最终评估不依赖生成的簇名称；只有指定 `--name-clusters` 才会额外请求命名并输出 `cluster_names.json`，这部分请求同样收费。

可以从保存的模型与中心重新评估，且不调用 GPT、不重新拟合 KMeans：

```bash
python run_mre.py --evaluate-run outputs/mre_seed0_替换为实际时间
```

复核时会验证 config 中保存的 manifest 指纹，以及已保存预测的样本 ID、标签和顺序；不匹配则报错。检查生成的 `reevaluation.json` 中 `predictions_identical` 是否为 `true`。该功能是推理复核，**不是精确断点续训**；当前没有保存完整优化器、调度器、采样器和全部随机状态以恢复中断训练。

对导师建议至少汇报：数据来源与 64/16 名单、传导式协议、各 seed 数据量、重复/歧义和截断审计、LLM 请求型号/API 返回版本/实际请求量、GPT 与无 GPT 的三项指标及均值/标准差。不要将簇名称看起来合理视为分类准确率正确的证据。

## 7. 实现依据与检查

样本协议、均衡采样和评估参考 [MRE_learn 的 `data_loader.py`](https://github.com/jyp0222/MRE_learn/blob/82afbeba49b411b8f0db5bee58f6fd8d87e7ac89/data_loader.py)、[`framework.py`](https://github.com/jyp0222/MRE_learn/blob/82afbeba49b411b8f0db5bee58f6fd8d87e7ac89/framework.py) 和 [`utils.py`](https://github.com/jyp0222/MRE_learn/blob/82afbeba49b411b8f0db5bee58f6fd8d87e7ac89/utils.py)。所核对版本中，MRE 的类别选择曾采用随机切分；本次根据已确认方案明确替换为原文件定义的固定 64/16 类，因此不是不加区分地复制其所有默认值。

上游 LOOP 基线来自 [Lackel/LOOP](https://github.com/Lackel/LOOP) 及本仓库修改前的 commit `7b139f4a5ed41cf52baead68d7a98b88571466ae`。新增入口的核心模块为 `mre_protocol.py`、`mre_data.py`、`mre_neighbors.py`、`mre_trainer.py`、`mre_metrics.py` 和 `llm_client.py`。

自动化测试位于 `tests/`，覆盖数据划分与标签隔离、指标与 MRE 参考实现的一致性、近邻逻辑及模拟 GPT-3.5 Chat Completions 请求；执行时不需要真实 API key：

```bash
python -m pytest tests -q
```

自动化检查不能替代服务器真实数据、GPU 训练及账号 API 可用性验证。不要把离线测试通过写成“已经跑出正式实验结果”。

初始协议版本的历史验证：103 项离线测试通过（协议 10、指标 6、数据加载 8、客户端 70、近邻/训练/复核 9），并通过 Python 3.8 语法检查。训练集成测试用随机初始化的单层小型 BERT 和合成的 2 base / 1 novel 数据，完成 CE+MLM、CE+RNCL、保存和重新加载，验证预测一致且评估不会调用 KMeans.fit；64/16 类别约束另有协议测试覆盖。当时训练测试环境为 Python 3.9、CPU PyTorch 2.8、Transformers 4.2.1。

本次 GPT-3.5 迁移验证：94 项客户端测试和 6 项无 BERT 的入口测试通过，修改文件通过 Python 3.8 语法及 diff 检查。系统 pytest 插件自动加载会干扰本地测试，关闭插件自动发现后通过。本次 BERT 集成测试在收集阶段被本地全局 Transformers 5.2.0 与 Python 3.9 的 `UnionType` 导入不兼容阻塞；没有修改服务器依赖，也没有重新验证 GPU 训练。上述历史训练结果不能代替本次服务器 smoke。本次未使用真实数据训练或发起付费 API 请求。
