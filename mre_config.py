"""Edit server paths and experiment settings here; no `export` is required."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = Path('/home/zhoubaohang/FewRel')  # original train.txt / test.txt
BERT_MODEL = Path('/home/zhoubaohang/jiangyipeng/LOOP/pretrained/bert-base-uncased')
TOKENIZER = BERT_MODEL
OUTPUT_ROOT = ROOT / 'outputs'
EXPERIMENT_VARIANT = 'loop_mre_fewrel_gpt35_v2'

# API key is requested invisibly on the first uncached call. Alternatively,
# point this to a PRIVATE text file outside this Git repository.
API_KEY_FILE = None
MODEL_NAME = 'gpt-3.5-turbo'  # available provider route; backend snapshot UNVERIFIED
NAMING_MODEL_NAME = 'gpt-3.5-turbo'  # same model for both relation selection and naming
REASONING_EFFORT = None  # GPT-3.5 has no reasoning mode; not sent to the API
API_BASE = 'https://api.zhizengzeng.com/v1'
MAX_OUTPUT_TOKENS = None  # upstream omits max_tokens; do not send JSON mode or reasoning options
REQUEST_TIMEOUT = 120
MAX_RETRIES = 0  # upstream makes one attempt before its observable Choice-1 fallback
MAX_REQUESTS = None  # cap actual HTTP attempts per run, including retries

SEED = 0
POSITION_FORMAT = 'indices'  # [13, 14] means TWO token positions
PRETRAIN_EPOCHS = 100
TRAIN_EPOCHS = 50
PATIENCE = 20  # pretraining ONLY; LOOP stage always completes TRAIN_EPOCHS
LABELED_BATCH_SIZE = 64
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 64
MAX_LENGTH = 128
LR_PRETRAIN = 5e-5
LR = 1e-5
WARMUP_PROPORTION = 0.1
GRAD_CLIP = 1.0
TEMPERATURE = 0.07
CE_WEIGHT = 0.5
TOPK = 50
UPDATE_EVERY = 5
QUERY_POOL_SIZE = 500  # intersect top-Q entropy and top-Q inconsistency
KMEANS_N_INIT = 10
VIEW_STRATEGY = 'rtr'
RTR_PROB = 0.25
NAME_CLUSTERS = True  # original post-training interpretation; never used by MRE metrics

# Intentionally absent: known_cls_ratio / labeled_ratio. The original source
# files define 64 base + 16 novel; the MRE sample protocol defines all labels.
