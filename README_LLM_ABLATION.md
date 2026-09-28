# GPT 温度与当前提示词

关系任务只使用 mre_prompts.py 中现有的提示词常量，不改提示词正文。
MET 继续使用同一文件中的实体类型提示词。训练、数据划分、caption、候选构造、
Choice 解析/回退、损失、模型选择和评估不变。

## 温度

--llm-temperature 默认 0.0，同时用于邻居选择和命名请求。
可在 mre_config.py 设置 LLM_TEMPERATURE = 0.0，或用命令行覆盖。
--llm-temperature provider 省略请求字段，仅用于复现历史服务端默认行为。
原 TEMPERATURE = 0.07 属于 RNCL，不要改成 0。

启动及 --check-llm 打印实际提示词版本和温度，运行配置和 API 日志也记录这些信息。
relation_prompt 审计字段保留为 current，兼容已有结果汇总脚本。
提示词选择参数已经删除，旧命令中的 --relation-prompt current 请去掉。
--experiment-variant 仅改变实验目录名称。

缓存 key 仍包含完整请求、端点和提示词版本；同样的输入及配置可读取已有缓存。
新运行目录不会自动复用其他目录的缓存。温度 0 能减少采样随机性，但不保证每次结果完全相同。
GPT 仍接收原始关系文本；caption 仅增强 BERT 输入。

## 命令

接口检查（一次付费请求，不训练）：

```bash
python -u run_mre.py --check-llm --task-type relation --llm-temperature 0
```

在现有 MRE 配置和 caption 缓存下运行：

```bash
for seed in 0 2 3; do
  CUDA_VISIBLE_DEVICES=0 python -u run_mre.py \
    --seed "$seed" --task-type relation \
    --use-image-caption --caption-path caption_cache/mg_mre_blip_base.json \
    --caption-model /home/zhoubaohang/jiangyipeng/LOOP-MRE-random40/pretrained/blip-image-captioning-base \
    --llm-temperature 0 \
    --experiment-variant mg_mre_caption_gpt35_t0_current \
    || break
done
```
