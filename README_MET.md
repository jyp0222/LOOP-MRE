# MET text-only entity type discovery

The explicit `TASK_TYPE = 'entity_type'` adapter accepts the supplied MET format:
`[sentence, image_url, topic, [[name, type, char_start, char_end, wiki_url], ...]]`.
It expands each entity annotation to one sample with a stable
`file:record_index:entity_index` ID. Positions are validated against the ORIGINAL
sentence as half-open character offsets. Mismatched names/offsets fail with the
file, record and annotation index, before writing prepared data. Whitespace is
normalized only after target markers are inserted.

Input example: `Entity: UK. Sentence: Students outside the [ENTITY] UK [/ENTITY] parliament.`
Only that entity is marked. Other mentions remain context. Entity type labels,
topic labels, image and Wikipedia URLs are never added to model/GPT input.
No images are opened or downloaded.

## Protocol

Pool train.json, valid.json, test.json in that order. Preserve first-appearance
class order, seeded NumPy class shuffle, then per-Base-class sample shuffle using
the same RNG as the existing MRE splitter. No optimizer/loss/batch/neighbor or
evaluation changes are made.

The current protocol filters `Other`, `None`, `none`, `NA`, including MET's
`Other`. Supplied inventory predicts:

| Source | Sentence records | Entity annotations | Filtered Other | Retained |
|---|---:|---:|---:|---:|
| train.json | 6312 | 13205 | 2307 | 10898 |
| valid.json | 755 | 1552 | 276 | 1276 |
| test.json | 757 | 1570 | 265 | 1305 |
| Total | 7824 | 16327 | 2848 | 13479 |

Twelve classes remain, so each seed yields 6 Base / 6 Novel. For each Base,
`n_train_val=floor(N*0.5)`, `n_train=floor(n_train_val*0.8)`; validation is the
difference. Remaining Base samples and all Novel samples form the test pool.
The same texts enter unlabeled training, with gold labels removed.

Splitting is by entity annotation, NOT sentence group. Different targets in one
sentence can cross partitions. The manifest reports overlap both by original
record ID and normalized unmarked sentence text; it does not silently change
the protocol to group splitting. Duplicates/conflicting labels are retained and
audited as before. This experiment is not the original supervised MET benchmark.

The legacy prepared-data field `relation` and audit fields `per_relation` refer
to entity types for MET; this preserves compatibility with existing consumers.

## Server configuration and checks

Update/add these fields in `mre_config.py` (replace existing fields, do not leave
duplicate assignments):

```python
TASK_TYPE = 'entity_type'
SOURCE_DIR = Path('/home/zhoubaohang/MG-VMoE/dataset/MET')
SOURCE_FILES = ('train.json', 'valid.json', 'test.json')
EXPECTED_CLASSES = 12
POSITION_FORMAT = 'half_open'
EXPERIMENT_VARIANT = 'loop_met_random6_cased_pre100_train50_bs64_64_64_v8'
```

Keep the current server's cased BERT/tokenizer, epochs 100/50, batch 64/64/64,
k=50, learning rates and API/key-file settings. The experiment name is metadata,
not an automatic hyperparameter preset. This patch does not edit the config file.

```bash
python -u run_mre.py --check-data --seed 0
python -u run_mre.py --check-llm
CUDA_VISIBLE_DEVICES=2 python -u run_mre.py --smoke --seed 0 --max-requests 20
```

Only the API check and an LLM smoke run may make paid calls. Smoke overrides
epochs to 2/2 and caps queried anchors to five per refresh. Check dataset counts,
truncation, GPU memory and fallback counts before running full experiments.
Real MET files are on the server; local tests use synthetic MET fixtures.

Neighbor and cluster-naming prompts switch together to entity type comparison:
`met-target-entity-type-choice-v1`. GPT model, candidate selection and Choice 1/2
parsing remain unchanged. Prompt versions isolate cache entries from relation
runs. Prompts include no gold class list. Existing offline neighbor audit works
with dynamic class counts; labels are used only for post-training diagnostics.

## Switch back

For FewRel or MG-VMoE MRE explicitly restore `TASK_TYPE = 'relation'` and all
dataset fields. FewRel: train.txt/test.txt, expected 80, indices; MG-VMoE MRE:
train.txt/val.txt/test.txt, expected 22, half_open. Restore the appropriate source
path and experiment name. Omitting TASK_TYPE defaults to relation for old configs.
The existing relation prompt is unchanged.
