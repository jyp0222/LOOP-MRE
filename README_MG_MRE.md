# MG-VMoE MRE: text-only discovery adaptation

This adapter pools `train.txt`, `val.txt`, `test.txt` in that explicit order.
It accepts JSON/Python-dictionary lines or a complete JSON array of dictionaries.
`token` and `tokens` are both accepted. Entity positions use `[start, end)`.
Images are not loaded. Relation labels remain unchanged and are never inserted
into input text or the LLM prompt.

The existing MRE discovery filter removes `Other`, `None`, `none`, `NA`.
The supplied dataset inventory therefore gives 15,485 source records minus
9,718 `None` records = 5,767 retained records in 22 relation classes:
train 4,503, val 624, test 640. These counts must be verified on the server.

The existing seeded random-half algorithm makes 11 Base and 11 Novel classes.
For each Base class, `floor(N*0.5)` records go to train+validation, of which
`floor(train_val*0.8)` are labeled training. Remaining Base records and all
Novel records form the test pool; its text also forms unlabeled training.
This is a new transductive discovery experiment, not evaluation under the
dataset's original supervised train/val/test split. Duplicates and conflicting
labels retain the existing report-only audit policy.

Add/update these fields in the server's `mre_config.py`:

```python
SOURCE_DIR = Path('/home/zhoubaohang/MG-VMoE/dataset/MRE')
SOURCE_FILES = ('train.txt', 'val.txt', 'test.txt')
EXPECTED_CLASSES = 22
POSITION_FORMAT = 'half_open'
EXPERIMENT_VARIANT = 'loop_mg_mre_random11_cased_pre1_train25_bs6_v6'
```

Keep the server's current BERT/tokenizer paths, pretraining 1 epoch, training
25 epochs, labeled/training batch sizes 6, API settings and other hyperparameters.
This patch deliberately does not edit the tracked config, trainer, neighbor
construction, losses, metrics or GPT prompts. The experiment name describes
these server settings; it does not set them automatically.

```bash
python -u run_mre.py --prepare-only --seed 0
python -u run_mre.py --check-data --seed 0
```

Check source counts, 11/11 classes and token truncation before training.
Prepared data, source hashes and configuration are saved separately for each run.
Then run the existing commands with seeds 0, 2, 3, with/without `--no-llm`.

Old FewRel defaults are preserved when the new fields are absent: train/test,
80 classes, and the existing position format. Switching back requires restoring
all five data/experiment fields (including `POSITION_FORMAT = 'indices'`).
The same fields can also be overridden using `--source-dir`, `--source-files`,
`--expected-classes`, `--position-format`, `--experiment-variant`.
