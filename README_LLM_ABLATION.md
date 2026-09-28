# GPT 温度与 MRE 提示词：分开验证

在当前 caption 分支上增加 LLM 配置，不修改 `mre_config.py` 的服务器路径、
超参数或现有提示词常量。`mre_trainer.py`、数据划分、caption 输入、候选构造、
Choice 解析/回退、损失、末轮模型选择和评估实现均不改。

## 参数

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--llm-temperature` | `0.0` | 所有 GPT 邻居/命名请求显式传入温度 0 |
| `--llm-temperature provider` | 手动选择 | 省略字段，复用历史服务端默认行为 |
| `--relation-prompt current` | `current` | 使用服务器 `mre_prompts.py` 中当前的原有提示词常量 |
| `--relation-prompt mre-v2` | 手动选择 | 使用新增的社交媒体有向关系提示词 |

温度可设置有限数值 0..2；服务器 `mre_config.py` 可以添加：

```python
LLM_TEMPERATURE = 0.0
RELATION_PROMPT = 'current'  # 或 'mre-v2'
```

不添加这两行也有上述默认值。原 `TEMPERATURE = 0.07` 属于 RNCL，勿改成 0。
`--experiment-variant` 仅改变实验标识/目录，不能切换提示词。
`current` 不等于强制恢复最初 FewRel 提示词：若服务器手动改过常量，它会保留该版本。
MET 继续使用自己的提示词，`entity_type` 与 `mre-v2` 组合直接报错。

启动、`--check-llm` 会打印提示词版本和温度。`config.json` 与 `llm_summary.json`
记录 `llm_temperature`，每条 `llm_calls.jsonl` 记录请求的 `temperature`。
完整请求（包括温度、实际消息文本）、端点和提示词版本共同决定缓存 key，
不会把历史省略温度的回答误用于温度 0，也不会混用不同提示词的回答。
缓存仍按原实现放在每个运行目录内；新目录不会自动复用以前目录的缓存。

## 新提示词

内容在 `mre_social_prompts.py`，版本 `mre-social-directed-relation-choice-v2`。
它是待验证的候选方案，不能预先称为效果最佳。

- 针对 Twitter 短文本、缩写、@mention、hashtag 和 RT 噪声。
- 关注句内 Head→Tail 的关系及角色，不按出现次序交换实体。
- 用局部上下文辨别具体关系，避免把同话题、同类型、共现当作同关系。
- 对稀疏文本不补写背景事实；仍保持二候选 `Choice 1/2`。
- 不提供真值关系标签、Base/Novel 名单；不新增弃权、置信度或多轮推理。

GPT 仍接收原始关系文本分词解码后的内容；caption 仅增强 BERT 输入。
temperature=0 只减少采样随机性；服务后端、数值计算或训练随机性仍可能造成差异。
新的提示词不能解决“两候选都不合适但必须选一个”的结构限制。

## 建议实验顺序

1. 当前提示词 + 温度 0，对照相同配置下历史省略温度的运行。
2. 新提示词 + 温度 0，对照第 1 组，仅改变提示词。

不要把“历史提示词/默认温度”与“新提示词/温度0”的差值归因于单独某项。
两个新组的 caption cache、seed、数据划分、BERT、batch、epoch、LR、评估保持一致。
先用 seed 0 跑 smoke，再按 0、2、3 配对运行；不能挑选重复运行中成绩最高的一次。
本修改不会影响 `--no-llm` 的模型计算路径。

服务器接口检查（一次付费请求，不训练）：

```bash
python -u run_mre.py --check-llm --task-type relation --llm-temperature 0 --relation-prompt mre-v2
```

在现有 MRE 配置、caption 缓存下，先检查训练链路：

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_mre.py \
  --smoke --seed 0 --max-requests 20 --task-type relation \
  --use-image-caption --caption-path caption_cache/mg_mre_blip_base.json \
  --caption-model /home/zhoubaohang/jiangyipeng/LOOP-MRE-random40/pretrained/blip-image-captioning-base \
  --llm-temperature 0 --relation-prompt mre-v2 \
  --experiment-variant mg_mre_caption_gpt35_t0_mrepromptv2_smoke
```

正式组 1（只将当前提示词的温度设为 0）：

```bash
for seed in 0 2 3; do
  CUDA_VISIBLE_DEVICES=0 python -u run_mre.py \
    --seed "$seed" --task-type relation \
    --use-image-caption --caption-path caption_cache/mg_mre_blip_base.json \
    --caption-model /home/zhoubaohang/jiangyipeng/LOOP-MRE-random40/pretrained/blip-image-captioning-base \
    --llm-temperature 0 --relation-prompt current \
    --experiment-variant mg_mre_caption_gpt35_t0_currentprompt \
    || break
done
```

正式组 2 用同一命令，将 `--relation-prompt current` 改为 `--relation-prompt mre-v2`，
实验名改为 `mg_mre_caption_gpt35_t0_mrepromptv2`。确认输出盘空间充足；需要换盘时给两组
都添加服务器实际可写的 `--output-root`。不要通过改 batch 等训练参数解决输出盘空间问题。
