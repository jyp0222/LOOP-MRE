"""Edit server paths and experiment settings here; no `export` is required."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = Path('/home/zhoubaohang/FewRel')  # original train.txt / test.txt
BERT_MODEL = Path('/home/zhoubaohang/jiangyipeng/LOOP/pretrained/bert-base-uncased')
TOKENIZER = BERT_MODEL
OUTPUT_ROOT = ROOT / 'outputs'

# API key is requested invisibly on the first uncached call. Alternatively,
# point this to a PRIVATE text file outside this Git repository.
API_KEY_FILE = None
MODEL_NAME = 'gpt-5.6-sol'  # alternative: 'gpt-6-astra'
REASONING_EFFORT = None  # None -> Sol: none; Astra: low
API_BASE = 'https://api.openai.com/v1'
MAX_OUTPUT_TOKENS = 4096
REQUEST_TIMEOUT = 120
MAX_RETRIES = 2
MAX_REQUESTS = None  # cap actual HTTP attempts per run, including retries

SEED = 0
POSITION_FORMAT = 'indices'  # [13, 14] means TWO token positions
PRETRAIN_EPOCHS = 100
TRAIN_EPOCHS = 50
PATIENCE = 10
LABELED_BATCH_SIZE = 16
TRAIN_BATCH_SIZE = 24
EVAL_BATCH_SIZE = 32
MAX_LENGTH = 128
LR_PRETRAIN = 5e-5
LR = 1e-5
WARMUP_PROPORTION = 0.1
GRAD_CLIP = 1.0
TEMPERATURE = 0.07
CE_WEIGHT = 0.5
TOPK = 20
UPDATE_EVERY = 5
QUERY_POOL_SIZE = 500  # intersect top-Q entropy and top-Q inconsistency
KMEANS_N_INIT = 10
NAME_CLUSTERS = False  # optional interpretation; never used by MRE metrics

# Intentionally absent: known_cls_ratio / labeled_ratio. The original source
# files define 64 base + 16 novel; the MRE sample protocol defines all labels.
