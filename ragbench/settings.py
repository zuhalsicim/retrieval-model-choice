"""Every fixed choice of the benchmark, in one place."""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Retrieval data: the test split of GermanDPR (deepset), CC BY 4.0.
# The repository's own loader script points to a zip file on S3 that answers
# "404 Not Found" (checked on 25 Nov 2026). The only copy left on the Hub is
# the Parquet conversion that Hugging Face made of it, on the branch
# refs/convert/parquet. The benchmark reads that file at a fixed commit and
# checks its SHA-256 before it uses it.
GERMANDPR_REPO = "deepset/germandpr"
GERMANDPR_CARD_REVISION = "5129d02422a66be600ac89cd3e8531b4f97d347d"  # main: card and loader script
GERMANDPR_PARQUET_REVISION = "d0b867a49c45a44d172041351b5df73a87714372"  # refs/convert/parquet
GERMANDPR_TEST_FILE = "plain_text/test/0000.parquet"
GERMANDPR_TEST_SHA256 = "f6106a22d6d6a6cab1621d70a06e0edd8edea9d1428e52424b1610f83b6228f0"

# Where the derived files live. They are small and part of the repository,
# so that the exact test data is versioned together with the code.
DATA_DIR = "data/germandpr-test"

# ---------------------------------------------------------------------------
# Prompt-injection data: deepset/prompt-injections, 546 train and 116 test rows.
INJECTION_REPO = "deepset/prompt-injections"
INJECTION_REVISION = "4f61ecb038e9c3fb77e21034b22511b523772cdd"
INJECTION_FILES = {
    "train": "data/train-00000-of-00001-9564e8b05b4757ab.parquet",
    "test": "data/test-00000-of-00001-701d16158af87368.parquet",
}


@dataclass(frozen=True)
class Embedder:
    key: str  # short name used in file names and tables
    repo: str
    revision: str  # a commit of the model repository on Hugging Face
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 512
    params_millions: int = 0  # from the model card, for the table only


@dataclass(frozen=True)
class Reranker:
    key: str
    repo: str
    revision: str
    max_length: int = 512
    params_millions: int = 0
    queries: int = 300  # how many of the RERANK_QUERIES it reranks (a nested random subset)


@dataclass(frozen=True)
class Guard:
    key: str
    repo: str
    revision: str
    injection_label: str  # the label the model gives to an injection
    params_millions: int = 0
    extra: dict = field(default_factory=dict)


EMBEDDERS = [
    Embedder(
        "e5-small",
        "intfloat/multilingual-e5-small",
        "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        query_prefix="query: ",
        passage_prefix="passage: ",
        params_millions=118,
    ),
    Embedder(
        "e5-base",
        "intfloat/multilingual-e5-base",
        "d128750597153bb5987e10b1c3493a34e5a4502a",
        query_prefix="query: ",
        passage_prefix="passage: ",
        params_millions=278,
    ),
    Embedder(
        "minilm-paraphrase",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        max_seq_length=128,  # the model's own limit, set in its sentence_bert_config.json
        params_millions=118,
    ),
    Embedder(
        "granite-278m",
        "ibm-granite/granite-embedding-278m-multilingual",
        "a9cb5338491faf32b73dd17b714a31821c021bbf",
        params_millions=278,
    ),
    Embedder(
        "bge-m3",
        "BAAI/bge-m3",
        "5617a9f61b028005a4858fdac845db406aefb181",
        params_millions=568,
    ),
]

RERANKERS = [
    Reranker(
        "bge-reranker-v2-m3",
        "BAAI/bge-reranker-v2-m3",
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        params_millions=568,
        queries=100,  # the largest model gets fewer queries, to keep the run short
    ),
    Reranker(
        "mmarco-minilm",
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "1427fd652930e4ba29e8149678df786c240d8825",
        params_millions=118,
    ),
]

GUARDS = [
    Guard(
        "protectai-v2",
        "protectai/deberta-v3-base-prompt-injection-v2",
        "90c9989b1a342275dd0d1a95aad283c04e075671",
        injection_label="INJECTION",
        params_millions=184,
    ),
]

# ---------------------------------------------------------------------------
# How retrieval is measured.
SEED = 42
RETRIEVE_DEPTH = 100  # passages kept per query, enough for Recall@100
ENCODE_BATCH_SIZE = 32
LATENCY_QUERIES = 200  # queries timed one by one, as a live system would see them

# Reranking: the first stage is the best embedding model whose licence is
# clear for a product ("yes" in ragbench/licences.py), chosen from the
# result files, not by hand. Its top RERANK_DEPTH passages are reordered.
RERANK_DEPTH = 20
RERANK_QUERIES = 300  # a fixed random subset (SEED) of the 1025 test queries
RERANK_BATCH_SIZE = 20

# Quantization: binary search keeps this many candidates for rescoring.
BINARY_RESCORE_DEPTH = 40

# Statistics.
BOOTSTRAP_SAMPLES = 10_000
PERMUTATIONS = 10_000

# The latency target of the decision summary: one query, embedding plus
# search plus reranking, at the 95th percentile, on this CPU.
LATENCY_TARGET_MS = 500

# The smoke run in CI: a few queries against the passages they need.
SMOKE_QUERIES = 8
SMOKE_EMBEDDER = "e5-small"
SMOKE_RERANKER = "mmarco-minilm"


def embedder(key: str) -> Embedder:
    return next(e for e in EMBEDDERS if e.key == key)


def reranker(key: str) -> Reranker:
    return next(r for r in RERANKERS if r.key == key)
