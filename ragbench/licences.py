"""The licence check: what each model card and dataset card says at the pinned revision.

`commercial` is the verdict for use in a product:
- "yes": the licence allows commercial use and the card names nothing that restricts it.
- "check": the licence allows commercial use, but the card itself says the model was trained on
  data whose own terms are non-commercial (MS MARCO). Whether that binds the model is a legal
  question this benchmark cannot answer; it has to go to legal review before product use.
- "no": the licence forbids commercial use. Such a model is out, however well it scores.

`python -m ragbench licences` fetches every card again at its revision and checks that the
licence field is still what this table says (results/licence-check.json).
"""

import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

MS_MARCO_TERMS = (
    "The MS MARCO datasets are intended for non-commercial research purposes only "
    "(https://microsoft.github.io/msmarco/, section Terms and Conditions, read 25 Nov 2026)"
)

ENTRIES = [
    # Embedding models
    {
        "repo": "intfloat/multilingual-e5-small",
        "kind": "model",
        "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "card_licence": "mit",
        "commercial": "check",
        "measured": "yes",
        "note": "The card lists MS MARCO (500k pairs) in its supervised fine-tuning data. " + MS_MARCO_TERMS + ".",
        "quote": "| [MS MARCO](https://microsoft.github.io/msmarco/) | English | 500k |",
    },
    {
        "repo": "intfloat/multilingual-e5-base",
        "kind": "model",
        "revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
        "card_licence": "mit",
        "commercial": "check",
        "measured": "yes",
        "note": "Same training data table as e5-small, with MS MARCO.",
        "quote": "| [MS MARCO](https://microsoft.github.io/msmarco/) | English | 500k |",
    },
    {
        "repo": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "kind": "model",
        "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        "card_licence": "apache-2.0",
        "commercial": "yes",
        "measured": "yes",
        "note": "The card does not name its training data. A paraphrase model, not trained for search.",
        "quote": "This model was trained by [sentence-transformers](https://www.sbert.net/).",
    },
    {
        "repo": "ibm-granite/granite-embedding-278m-multilingual",
        "kind": "model",
        "revision": "a9cb5338491faf32b73dd17b714a31821c021bbf",
        "card_licence": "apache-2.0",
        "commercial": "yes",
        "measured": "planned; download failed",
        "note": "The card lists its training data and says it leaves MS MARCO out because of its licence.",
        "quote": "we do not use the popular MS-MARCO retrieval dataset in our training corpus "
        "due to its non-commercial license",
    },
    {
        "repo": "BAAI/bge-m3",
        "kind": "model",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "card_licence": "mit",
        "commercial": "yes",
        "measured": "planned; download failed",
        "note": "The card links its fine-tuning data (Shitao/bge-m3-data); that data was not audited here.",
        "quote": "license: mit",
    },
    {
        "repo": "Alibaba-NLP/gte-multilingual-base",
        "kind": "model",
        "revision": "6dde5e2ebcac02a5459a3cf310346901fa1dc8ff",
        "card_licence": "apache-2.0",
        "commercial": "yes",
        "measured": "no",
        "note": "Not measured, for time. At this revision it no longer needs remote code (model_type gte).",
        "quote": "license: apache-2.0",
    },
    {
        "repo": "jinaai/jina-embeddings-v2-base-de",
        "kind": "model",
        "revision": "3f9eede875721714945b6a99a3198299243cf2be",
        "card_licence": "apache-2.0",
        "commercial": "yes",
        "measured": "no",
        "note": "Not measured: it loads Python code from another repository (jinaai/jina-bert-implementation) "
        "with trust_remote_code, which this benchmark does not run without a code review.",
        "quote": '"AutoModel": "jinaai/jina-bert-implementation--modeling_bert.JinaBertModel"',
    },
    {
        "repo": "jinaai/jina-embeddings-v3",
        "kind": "model",
        "revision": "ab036b023d30b4d1138c4c3bfa9f0c445ab455d6",
        "card_licence": "cc-by-nc-4.0",
        "commercial": "no",
        "measured": "no",
        "note": "Non-commercial. Out for a product, also when run on own hardware.",
        "quote": "If you need to use it beyond those platforms or on-premises within your company, "
        "note that the models is licensed under CC BY-NC 4.0.",
    },
    # Rerankers
    {
        "repo": "BAAI/bge-reranker-v2-m3",
        "kind": "model",
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "card_licence": "apache-2.0",
        "commercial": "yes",
        "measured": "planned; download failed",
        "note": "The card does not name its training data.",
        "quote": "license: apache-2.0",
    },
    {
        "repo": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "kind": "model",
        "revision": "1427fd652930e4ba29e8149678df786c240d8825",
        "card_licence": "apache-2.0",
        "commercial": "check",
        "measured": "planned; download failed",
        "note": "Trained on mMARCO, a machine translation of MS MARCO. " + MS_MARCO_TERMS + ".",
        "quote": "This model was trained on the [MMARCO](https://hf.co/unicamp-dl/mmarco) dataset. "
        "It is a machine translated version of MS MARCO using Google Translate.",
    },
    {
        "repo": "jinaai/jina-reranker-v2-base-multilingual",
        "kind": "model",
        "revision": "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9",
        "card_licence": "cc-by-nc-4.0",
        "commercial": "no",
        "measured": "no",
        "note": "Non-commercial, and it needs trust_remote_code. Screened out before any compute was spent.",
        "quote": "This model repository is licenced for research and evaluation purposes under CC-BY-NC-4.0.",
    },
    # Prompt-injection classifiers
    {
        "repo": "protectai/deberta-v3-base-prompt-injection-v2",
        "kind": "model",
        "revision": "90c9989b1a342275dd0d1a95aad283c04e075671",
        "card_licence": "apache-2.0",
        "commercial": "yes",
        "measured": "planned; download failed",
        "note": "English only by its own card, and the project is archived (no further updates).",
        "quote": "It does not detect jailbreak attacks or handle non-English prompts",
    },
    {
        "repo": "deepset/deberta-v3-base-injection",
        "kind": "model",
        "revision": "80dda00d0b0d9a03917a7685e2ddbcd28e04dbb1",
        "card_licence": "mit",
        "commercial": "yes",
        "measured": "no",
        "note": "Not measured: it was trained on deepset/prompt-injections, the evaluation data here. "
        "Its card reports accuracy 0.9914 on its evaluation set, which is 115 of 116, the size of the "
        "test split; so the test split was seen during training and gives no independent figure.",
        "quote": "It achieves the following results on the evaluation set: - Loss: 0.0673 - Accuracy: 0.9914",
    },
    {
        "repo": "meta-llama/Llama-Prompt-Guard-2-86M",
        "kind": "model",
        "revision": "a8ded8e697ce7c355e395a0df51f94adb4a2fd27",
        "card_licence": "other (llama4)",
        "commercial": "check",
        "measured": "no",
        "note": "Gated: card and weights need an account that accepts Meta's licence first. Not done here.",
        "quote": "(card not readable without accepting the licence; metadata: license other, license_name llama4)",
    },
    # Data
    {
        "repo": "deepset/germandpr",
        "kind": "dataset",
        "revision": "5129d02422a66be600ac89cd3e8531b4f97d347d",
        "card_licence": "cc-by-4.0",
        "commercial": "yes",
        "measured": "yes",
        "note": "Attribution required. The card's loader points to a zip on S3 that returns 404; "
        "the data comes from the Hub's Parquet conversion (refs/convert/parquet).",
        "quote": "license: cc-by-4.0",
    },
    {
        "repo": "deepset/prompt-injections",
        "kind": "dataset",
        "revision": "4f61ecb038e9c3fb77e21034b22511b523772cdd",
        "card_licence": "apache-2.0",
        "commercial": "yes",
        "measured": "yes",
        "note": "The card states two licences: apache-2.0 at the top, cc-by-4.0 inside dataset_info. "
        "Both allow commercial use; CC BY 4.0 adds attribution.",
        "quote": "dataset_info: ... license: cc-by-4.0 / license: apache-2.0",
    },
]


def card_metadata(repo: str, kind: str, revision: str) -> dict:
    url = f"https://huggingface.co/api/{'datasets' if kind == 'dataset' else 'models'}/{repo}/revision/{revision}"
    request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(request, timeout=60) as response:
        info = json.loads(response.read())
    card = info.get("cardData") or {}
    return {
        "sha": info.get("sha"),
        "license": card.get("license"),
        "license_name": card.get("license_name"),
        "gated": info.get("gated"),
    }


def check(out: Path = Path("results/licence-check.json")) -> list[dict]:
    rows = []
    for entry in ENTRIES:
        fetched = card_metadata(entry["repo"], entry["kind"], entry["revision"])
        licence = fetched["license"]
        if isinstance(licence, list):
            licence = ",".join(licence)
        if fetched["license_name"]:
            licence = f"{licence} ({fetched['license_name']})"
        rows.append(
            {
                "repo": entry["repo"],
                "revision": entry["revision"],
                "revision_found": fetched["sha"] == entry["revision"],
                "licence_in_table": entry["card_licence"],
                "licence_on_card": licence,
                "gated": fetched["gated"],
                "same": licence == entry["card_licence"],
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"checked_at": datetime.now(UTC).isoformat(timespec="seconds"), "rows": rows}
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return rows
