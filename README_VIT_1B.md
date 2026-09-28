# Experiment 1B：冻结 ViT，仅融合最终测试聚类特征

以当前 LOOP-MRE 实现为 baseline。默认关闭，不删除 Experiment 1A。
1B 与 caption 不能同时开启。无联合训练、投影、attention/gating 或 hyperbolic 模块。

## 数据与特征流

```
原始 train/val/test -> 当前过滤和随机半类协议 -> 当前纯文本 BERT 训练
                                                  |
                                  最后一个 epoch 的 BERT CLS [N,768]
                                                  | L2 normalize
                                                  +-------------------+
                                                                      | concat
img_id -> image_root/img_id -> 冻结 ViT CLS [unique_images,768]          | [N,1536]
                            -> 离线缓存 -> 按测试 sample_id 对齐        |
                                       -> L2 normalize ---------------+
                                                                      |
                                                      当前测试 KMeans + MRE 评估
```

最终拼接不再次归一化，不额外增加可训练参数。d_image 读取缓存维度，KMeans 动态处理维度。
本轮的训练期图聚类、nearest neighbors、LIS、LLM 候选/请求、RNCL、
每五轮的中间测试以及 post-test 命名仍使用文本特征。只有末轮最终测试/对应重评估融合。
RNCL 使用原有 128 维 head；聚类的文本分支使用原有 768 维 backbone CLS，二者不同。

当前指标实际是 Base / Novel / Overall，使用原有全局 Hungarian 匹配。
没有新增 H-score 或单独子集匹配；现有汇总和 mean/std 脚本仍可读取 results.json。

## ViT 和预处理

默认：google/vit-base-patch16-224-in21k，可换成本地 HF 模型目录。
使用 last_hidden_state[:, 0, :]，即最后一层归一化后的 CLS，而非分类 logits、pooler 或 patch 平均。
模型卡说明 CLS 可作整图表示：https://huggingface.co/google/vit-base-patch16-224-in21k
默认 checkpoint 为 224x224 / patch16 / hidden768。

使用 checkpoint 自带 processor 的 resize、rescale 和 RGB normalization，
不自行指定新的图像增强；先做 EXIF 方向校正并转 RGB。
兼容旧 transformers 的 ViTFeatureExtractor 和新版本的 ViTImageProcessor；不依赖 timm。
模型 eval()、requires_grad_(False)，推理 no_grad()。模型只在离线生成脚本中加载。
本机安装情况不代表服务器；服务器可先检查：

```bash
python -c "import transformers, torch; from transformers import ViTModel; print(transformers.__version__, torch.__version__)"
```

本地模型目录须有 config.json、preprocessor_config.json 和可被当前 transformers
加载的权重（旧环境建议 pytorch_model.bin）。不把权重或视觉缓存加入 Git。

## 参数

| 参数 | 默认/用途 |
| --- | --- |
| --use-image-feature / --use_image_feature | 默认 False，启用 1B |
| --no-image-feature | 强制关闭视觉特征，恢复原始聚类输入 |
| --image-encoder / --image_encoder | vit |
| --image-model-path / --image_model_path | HF ID 或本地目录 |
| --freeze-image-encoder / --freeze_image_encoder | 默认 True；本实验拒绝不冻结 |
| --image-feature-cache / --image_feature_cache | 离线缓存目录，开启时必填 |
| --image-root | 图像根目录，可沿用缓存记录的目录 |
| --image-field | img_id |

同名 Python 配置常量可写入 mre_config.py，但本提交不改你的服务器配置。

## 离线缓存（只做一次，所有 seed 共享）

预先将默认 ViT 下载到下面的本地目录，然后执行；没有模型时可用 HF ID 替换该路径，
由 preprocessing 脚本下载，训练本身不会下载或加载 ViT。

```bash
cd /home/zhoubaohang/jiangyipeng/LOOP-MRE-random40

python -u prepare_image_features.py \
  --source-dir /home/zhoubaohang/MG-VMoE/dataset/MRE \
  --source-files train.txt val.txt test.txt --position-format half_open \
  --image-root /home/zhoubaohang/MG-VMoE/dataset/MRE/image \
  --check-images

CUDA_VISIBLE_DEVICES=0 python -u prepare_image_features.py \
  --source-dir /home/zhoubaohang/MG-VMoE/dataset/MRE \
  --source-files train.txt val.txt test.txt --position-format half_open \
  --image-root /home/zhoubaohang/MG-VMoE/dataset/MRE/image \
  --image-model-path /home/zhoubaohang/jiangyipeng/LOOP-MRE-random40/pretrained/vit-base-patch16-224-in21k \
  --image-feature-cache image_feature_cache/mg_mre_vit_cls \
  --device cuda:0 --batch-size 16
```

这里的 batch-size 仅用于离线 ViT 推理，不改变 LOOP batch。
缓存目录包含 manifest.json 和 vectors/<hash(image_id)>.npy，每个唯一图像一条 FP32 向量。
manifest 绑定源文件 hash、sample→image 映射、模型/预处理/CLS 信息、
本地权重 hash、每张图像 hash、每个向量 hash；第二次生成直接复用，支持中断后续算。
并发训练可读同一缓存；不要让两个 preprocessing 进程写同一个缓存。

运行启动时检查并保存 image_features.npz（最终测试样本顺序）、image_feature_audit.json。
快照内不存标签。重新评估只需要运行目录，不依赖原图片、原缓存或 ViT。
启动打印三条 sample 的文本、图像路径和预期维度；最终聚类时打印实际维度，
并保存 final_feature_shapes.json。默认 ViT 的测试快照约为 N*768*4 字节（压缩前）。

## Baseline 与 1B（相同训练设置，caption 均关闭）

先用 seed 0 smoke；下面默认使用服务器现有 MRE source/BERT/超参配置。
若当前配置是 MET/FewRel，需要明确选择 MRE，不能仅改 experiment-variant 名字。

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_mre.py \
  --smoke --seed 0 --max-requests 20 \
  --task-type relation --source-dir /home/zhoubaohang/MG-VMoE/dataset/MRE \
  --source-files train.txt val.txt test.txt --expected-classes 22 --position-format half_open \
  --no-image-caption --use-image-feature --freeze-image-encoder \
  --image-model-path /home/zhoubaohang/jiangyipeng/LOOP-MRE-random40/pretrained/vit-base-patch16-224-in21k \
  --image-feature-cache image_feature_cache/mg_mre_vit_cls \
  --llm-temperature 0 --experiment-variant mg_mre_exp1b_vit_cls_smoke
```

纯文本 baseline：

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_mre.py \
  --seed 0 --task-type relation \
  --source-dir /home/zhoubaohang/MG-VMoE/dataset/MRE \
  --source-files train.txt val.txt test.txt --expected-classes 22 --position-format half_open \
  --no-image-caption --no-image-feature --llm-temperature 0 \
  --experiment-variant mg_mre_exp1b_text_baseline
```

ViT 实验：

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_mre.py \
  --seed 0 --task-type relation \
  --source-dir /home/zhoubaohang/MG-VMoE/dataset/MRE \
  --source-files train.txt val.txt test.txt --expected-classes 22 --position-format half_open \
  --no-image-caption --use-image-feature --freeze-image-encoder \
  --image-model-path /home/zhoubaohang/jiangyipeng/LOOP-MRE-random40/pretrained/vit-base-patch16-224-in21k \
  --image-feature-cache image_feature_cache/mg_mre_vit_cls \
  --llm-temperature 0 --experiment-variant mg_mre_exp1b_vit_cls
```

两个命令都可添加 --no-llm，分别做无 GPT 配对；正式 seed 为 0/2/3。
--evaluate-run <已保存的1B运行目录> 自动使用其快照重新评估；不要用参数改变旧运行含义。

## 对齐、维度与解释边界

- sample id 仍为原 parser 的文件 stem + 原始数组/行号；同图多样本复用同一视觉向量。
- img_id 原样拼接图像根目录，不猜扩展名或递归搜索；缺失/不可读图、缺失向量、
  变更图片/源数据、损坏缓存、顺序错配都报错，不删除样本或静默填零。
- 无图像选项时不读视觉缓存/图片，不归一化原始聚类文本 CLS，完全走原特征路径。
- model.py 的预训练 classifier 仍固定 768，RNCL head 仍128；这些都是当前文本训练限制，
  不处理融合特征，也无需为 1536 维加投影。训练图、命名仍768维，最终聚类 centers 为1536维。
- 在原始文本 CLS baseline 上加视觉的同时，按本实验要求归一化文本分支，
  因此结果差值包含文本归一化的影响；不能全部归因于视觉。后续可另做 text-L2-only 对照，
  本轮不擅自加入该实验。
- 1A 会改变文本训练表示；1B 仅改变最终聚类输入，两者注入视觉信息的位置不同。
- GPT 温度0仍不能保证服务响应一致。相同 seed 的训练路径不变，但独立重跑仍可能有后端/GPU噪声。
