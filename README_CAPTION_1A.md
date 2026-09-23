# Experiment 1A: Image Caption Augmented Text (MRE only)

Baseline: current modified LOOP-MRE at `c1a9314`, not original upstream LOOP.
One main variable: append ` The image caption is: ` and an offline image caption
to the existing directed Head/Tail/marked-sentence input before BERT tokenization.
No image-feature fusion, new loss, sampling strategy or metric is introduced.

## Data and isolation

The MG-VMoE MRE examples supplied by the user use `token`, `h`, `t`, `relation`,
and `img_id` (e.g. `twitter_19_31_16_6.jpg`). `img_id` is resolved verbatim under
the explicitly supplied image root; no `.jpg` is added and no recursive match is
guessed. Actual server image availability must be checked with the command below.
The current adapter had no caption field/implementation and discarded image IDs.
An optional side channel in the same parser now retrieves them without changing
baseline records, filtering, first-appearance class order or line/array IDs.

Source files remain train.txt + val.txt + test.txt in that order, with half-open
token spans. Existing None/Other/none/NA filtering remains unchanged. For this
dataset, the observed 5,767 retained samples / 22 relations become random 11/11
classes. Base sample proportions remain approximately 40% labeled / 10% validation
/ 50% test; all Novel samples enter test; test text is also the unlabeled pool.
Caption extraction covers all retained samples including validation/test, using
images only: it never sends relation labels, sentence text or entity names into
the caption model. Each unique image is generated once and shared across seeds.

Original prepared JSONL files and manifest are unchanged. Each augmented run gets
its own `caption_inputs.json` outside `data/`, with original text, image ID/hash,
caption and encoder text per sample ID. SHA256 checks bind it to config.json.
Saved-run evaluation reads this snapshot, not the mutable external cache or images.
All splits get the same input augmentation; test/unlabeled tensors still share
inputs. Labels and sample IDs are never changed.

**GPT still receives the same original tokenizer-visible text as baseline.**
The two existing decode sites (neighbor query and post-test cluster naming) use
original text tokens when captions are enabled. Prompt, model, cache policy,
candidate construction, LIS, RNCL, KMeans and evaluation are unchanged. Features
will change, so selected neighbors/GPT requests may naturally differ downstream.

## 1. Keep the experiment configuration paired

Do not replace your entire server `mre_config.py` with tracked defaults. Keep your
working API settings/key path and training settings. For the new MRE dataset set:

```python
SOURCE_DIR = Path('/home/zhoubaohang/MG-VMoE/dataset/MRE')
TASK_TYPE = 'relation'
SOURCE_FILES = ('train.txt', 'val.txt', 'test.txt')
EXPECTED_CLASSES = 22
POSITION_FORMAT = 'half_open'
BERT_MODEL = Path('/home/zhoubaohang/jiangyipeng/LOOP-MRE-random40/pretrained/bert-base-cased')
TOKENIZER = BERT_MODEL
```

For a pair based on your completed MRE v7 experiment: pretrain maximum 100,
train 50, labeled/train/eval batch 64/64/64, k=50, MAX_LENGTH=128. Keep the same
learning rates, RTR, LLM settings and all other existing settings for both runs.
If your config still has MET's train batch 128, decide which baseline you intend
to compare with before running; do not compare a 128 run to the historical 64 run.
Pretraining's existing validation selection/early stopping is retained; actual
pretraining epoch count can differ as a consequence of the changed input.

Optional config values (all also available on the command line):

```python
USE_IMAGE_CAPTION = False
CAPTION_PATH = ROOT / 'caption_cache' / 'mg_mre_blip_base.json'
CAPTION_MODEL = str(ROOT / 'pretrained' / 'blip-image-captioning-base')
CAPTION_IMAGE_FIELD = 'img_id'
```

Defaults without these additions: disabled, no cache path,
`Salesforce/blip-image-captioning-base`, `img_id`. `--no-image-caption` explicitly
overrides a True config value. Underscore aliases exist for `--use_image_caption`,
`--caption_path`, `--caption_model`. No max_length change is needed.

## 2. Offline generation in a separate environment

Use a separate environment because the original Python 3.8 / old Transformers
training environment may not include BLIP. This does not upgrade `loop-mre`.
The default captioner is the commonly used pretrained
[Salesforce BLIP image-captioning base](https://huggingface.co/Salesforce/blip-image-captioning-base).
It uses image-only greedy decoding (`do_sample=False`, `num_beams=1`, at most 40
new tokens). A BLIP-compatible local HuggingFace model directory is supported.
Its internal image encoder is used only to produce a caption; no visual feature
is passed into LOOP.

```bash
cd /home/zhoubaohang/jiangyipeng/LOOP-MRE-random40
conda create -n loop-caption python=3.10 -y
conda activate loop-caption
python -m pip install torch==2.2.2 transformers==4.40.2 huggingface-hub==0.23.5 numpy==1.26.4 Pillow==10.4.0

python -u prepare_image_captions.py --check-images --source-dir /home/zhoubaohang/MG-VMoE/dataset/MRE --image-root /home/zhoubaohang/MG-VMoE/dataset/MRE/image
```

Download the model once on a machine with HuggingFace access; if the server
cannot connect, transfer the whole downloaded directory there. Do not download
or replace the BERT weights for this experiment.

```bash
huggingface-cli download Salesforce/blip-image-captioning-base --local-dir pretrained/blip-image-captioning-base

python -u prepare_image_captions.py --source-dir /home/zhoubaohang/MG-VMoE/dataset/MRE --image-root /home/zhoubaohang/MG-VMoE/dataset/MRE/image --caption-model /home/zhoubaohang/jiangyipeng/LOOP-MRE-random40/pretrained/blip-image-captioning-base --caption-path caption_cache/mg_mre_blip_base.json --device cpu
```

CPU generation avoids occupying your training GPU but is slower. If GPU 2 is
available, the same preprocessing command can be prefixed with
`CUDA_VISIBLE_DEVICES=2` and use `--device cuda:0`. Freeze the resulting one cache
for all paired seeds. `--revision` can pin a remote model commit; local model files
are fingerprinted. Cache generator settings and source-image mapping must match.

Interrupted preprocessing resumes from completed images. A complete cache causes
no caption model load. Do not run two generation writers against the same cache.
Missing/corrupt images, absent/blank captions, altered model/image hashes or a cache
from different source files fail explicitly, without dropping samples. A network
download failure does not change any training data.

## 3. Seed 0 checks and paired smoke tests

After setting the MRE config above, return to the unchanged training environment:

```bash
conda activate loop-mre
python -u run_mre.py --check-data --seed 0 --no-image-caption
python -u run_mre.py --check-data --seed 0 --use-image-caption --caption-path caption_cache/mg_mre_blip_base.json --caption-model /home/zhoubaohang/jiangyipeng/LOOP-MRE-random40/pretrained/blip-image-captioning-base

CUDA_VISIBLE_DEVICES=2 python -u run_mre.py --smoke --seed 0 --no-llm --no-image-caption --experiment-variant mg_mre_exp1a_smoke_baseline
CUDA_VISIBLE_DEVICES=2 python -u run_mre.py --smoke --seed 0 --no-llm --use-image-caption --caption-path caption_cache/mg_mre_blip_base.json --caption-model /home/zhoubaohang/jiangyipeng/LOOP-MRE-random40/pretrained/blip-image-captioning-base --experiment-variant mg_mre_exp1a_smoke_caption
```

These smoke tests intentionally disable GPT on **both** sides to first isolate
encoder augmentation without paid queries. For GPT-enabled checks, remove
`--no-llm` on **both** sides and add `--max-requests 20` on **both** smoke commands.
Do not compare GPT-on caption runs against GPT-off baselines.

Startup prints model/cache coverage and 3 reproducibly sampled examples (independent
Python Random instance, not training RNG). `tokenization_audit.json` includes
`original_truncated`, `truncated`, `newly_truncated_by_caption` and maximum lengths.
`unique_samples` counts each ID once; test is not double-counted with unlabeled.
Augmented text keeps the exact existing prefix/markers then appends the caption.

## 4. Formal paired experiment (same seed 0)

```bash
CUDA_VISIBLE_DEVICES=2 python -u run_mre.py --seed 0 --no-llm --no-image-caption --experiment-variant mg_mre_exp1a_baseline
CUDA_VISIBLE_DEVICES=2 python -u run_mre.py --seed 0 --no-llm --use-image-caption --caption-path caption_cache/mg_mre_blip_base.json --caption-model /home/zhoubaohang/jiangyipeng/LOOP-MRE-random40/pretrained/blip-image-captioning-base --experiment-variant mg_mre_exp1a_caption
```

For your GPT-enabled baseline use exactly this pair with `--no-llm` removed on
both commands. Repeat seeds 2 and 3 only after seed 0 succeeds. Reuse the same
caption cache. Metrics remain Base / Novel / Overall with the current global
Hungarian matching and test KMeans protocol. No H-score is added.

`compare_caption_runs.py --baseline <baseline run directory> --caption-run <caption run directory>`
checks recorded hyperparameters, source-code/package fingerprints, manifest and
split-file hashes. These arguments are the actual directories printed by the two
runs, not the literal angle-bracket text. For the no-GPT formal pair above you
can select the newest completed pair automatically:

```bash
python - <<'PY'
from pathlib import Path
from compare_caption_runs import compare_runs
def latest(prefix):
    runs = sorted(p for p in Path('outputs').glob(prefix + '_seed0_*') if (p / 'results.json').is_file())
    if not runs:
        raise RuntimeError('No completed run for ' + prefix)
    return runs[-1]
a, b = latest('mg_mre_exp1a_baseline'), latest('mg_mre_exp1a_caption')
print(a, b, sep='\n')
print(compare_runs(a, b))
PY
```

## Limits

No claim of improved accuracy before the experiment. Caption hallucinations,
irrelevant Twitter images, and generic captions may harm directed relation
features. Truncation can remove some/all of the appended caption. Keep max_length
128 for this ablation and inspect the audit before considering a separate length
experiment. Input augmentation changes gradients and downstream neighborhoods;
the paired run is not expected to have identical RNG trajectories or LLM calls.
Remote GPT responses and GPU numerics can also vary between repeats.

The local automated test suite exercises synthetic images, caching/resume,
mapping/split preservation, GPT original-text isolation and tiny-BERT training /
saved-run evaluation. It does not claim validation of the laboratory's real images
or an actual downloaded BLIP checkpoint; perform the server checks above.
