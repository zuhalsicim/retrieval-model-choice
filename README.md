# Retrieval model choice for German RAG on a CPU

Which open embedding model and which reranker should find German passages for an agent,
when the models must run on own hardware, must be usable in a product, and must answer fast?
This repository measures it: quality, licence, cost and speed, with the statistics to say which differences are real.
A smaller second part, ready but not yet run (see below), tests one open prompt-injection classifier on German and English inputs.

## Kurzfassung

- **Frage:** Welches offene Embedding-Modell findet deutsche Passagen am besten, wenn es auf eigener CPU laufen und in einem Produkt einsetzbar sein muss? Testdaten: GermanDPR (deepset), 1025 deutsche Fragen, 2876 Passagen, CC BY 4.0.
- **Gemessen** wurden drei Modelle. multilingual-e5-base erreicht nDCG@10 = 80,3, multilingual-e5-small 78,9, paraphrase-multilingual-MiniLM 64,1. Der Abstand zwischen e5-base und e5-small (+1,4 Punkte) ist mit diesen Daten nicht gesichert (95-%-Intervall −0,0 bis +2,9). MiniLM liegt klar zurück (16,2 Punkte hinter e5-base).
- **Kosten auf dieser CPU (Laptop, 8 Kerne):** e5-small kodiert 13,4 Passagen pro Sekunde, e5-base 3,5. Eine Suchanfrage dauert im Median 23 ms (e5-small) bzw. 58 ms (e5-base).
- **Lizenz entscheidet mit:** Beide e5-Modelle stehen unter MIT, ihre eigene Modellkarte nennt aber MS MARCO als Trainingsdaten, und MS MARCO ist laut Microsoft nur für nicht-kommerzielle Forschung gedacht. Vor einem Produkteinsatz muss das juristisch geprüft werden. jina-embeddings-v3 und jina-reranker-v2 sind CC BY-NC 4.0 und damit für ein Produkt ausgeschlossen. granite-embedding-278m (Apache-2.0) schließt MS MARCO laut Karte ausdrücklich aus.
- **Quantisierung:** int8-Vektoren sind 4-mal kleiner und kosten nichts Messbares, wenn die Anfrage als Float bleibt (80,3 → 80,3). Werden auch die Anfragen auf int8 gerundet, verliert e5-base 5,0 Punkte. Binäre Vektoren sind 32-mal kleiner; mit int8-Nachbewertung der besten 40 verliert e5-base 2,1 Punkte.
- **Nicht gemessen:** granite, bge-m3, beide Reranker und der Prompt-Injection-Klassifikator. Der Download der Gewichte brach ab, weil der Speicherserver von Hugging Face von diesem Rechner aus nicht erreichbar war. Der Code dafür ist fertig (`python -m ragbench all`).
- **Empfehlung unter Vorbehalt:** e5-small mit int8-Passagenvektoren und Float-Anfrage, falls die Rechtsprüfung Modelle mit MS MARCO im Training zulässt. Sonst als Nächstes granite-embedding-278m messen. Details unter „Entscheidungsgrundlage“.

## Results

Data: the 1025 questions of the GermanDPR test split against its 2876 passages. Every figure below comes from a file in `results/`; `python -m ragbench report` prints these tables.

### Retrieval: three embedding models

| Model | Params (M) | Licence for a product | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | Corpus encode (s) | Passages/s | One query p50 / p95 (ms) | Peak RAM (GB) | Dims | Index float32 (MB) | Weights (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e5-base | 278 | check | 80.3 | 98.6 | 100.0 | 74.3 | 814 | 3.5 | 58 / 74 | 2.67 | 768 | 8.8 | 1.11 |
| e5-small | 118 | check | 78.9 | 97.9 | 99.8 | 72.7 | 215 | 13.4 | 23 / 29 | 1.71 | 384 | 4.4 | 0.47 |
| minilm-paraphrase | 118 | yes | 64.1 | 84.3 | 96.3 | 57.6 | 110 | 26.1 | 34 / 54 | 1.35 | 384 | 4.4 | 0.47 |

Percentages. "check" and "yes" are explained under Licences. Files: `results/retrieval/*.json`.

| Comparison (A - B) | Metric | Queries | Difference (points) | 95 % interval, clustered | 95 % interval, per query | p (clustered) | Resolved? |
|---|---|---|---|---|---|---|---|
| e5-base - e5-small | ndcg@10 | 1025 | +1.4 | -0.0 to +2.9 | +0.3 to +2.5 | 0.0591 | no |
| e5-base - e5-small | recall@100 | 1025 | +0.2 | +0.0 to +0.5 | +0.0 to +0.5 | 0.5030 | no |
| e5-base - minilm-paraphrase | ndcg@10 | 1025 | +16.2 | +13.0 to +19.5 | +14.0 to +18.4 | 0.0001 | yes |
| e5-base - minilm-paraphrase | recall@100 | 1025 | +3.7 | +2.5 to +5.1 | +2.5 to +4.9 | 0.0001 | yes |

File: `results/comparisons.json`. A `p` of 0.0001 is the smallest value 10,000 random sign patterns can give (1 / 10,001).

What the numbers say, and what they do not say:

- **e5-base is not shown to be better than e5-small.** It is 1.4 points ahead in nDCG@10, but the clustered interval reaches 0 and the test gives p = 0.059. The interval that treats every question as independent (+0.3 to +2.5) would have called it a win. It is too narrow, because many questions share a passage (see Statistics).
- **e5-small costs much less.** On this CPU it encoded the corpus 3.8 times as fast (215 s against 814 s), answered one query in less than half the time (median 23 ms against 58 ms, p95 29 ms against 74 ms), held 1.71 GB of RAM against 2.67 GB, and its index is half the size.
- **The paraphrase model is the wrong tool for search.** paraphrase-multilingual-MiniLM is 16.2 points behind e5-base in nDCG@10, a difference far outside the noise. As its name says, it is a paraphrase model (no query and passage roles, unlike e5), and it reads only the first 128 tokens of a passage. Its card offers it for "clustering or semantic search"; on German question-to-passage search it is clearly weaker.
- **On this data the first stage almost never misses.** Recall@100 is 99.8 to 100 % for both e5 models: the relevant passage is nearly always among the first 100. The differences are in the order of the first ten. That is where a reranker would act; it could not be measured here (see below).
- **Speed was measured once, on one laptop.** It is not a statement about server CPUs. For one query, embedding the question takes nearly all the time; the search over 2876 passages took 1.9 to 3.3 ms (median).

### Quantization: smaller vectors, same models

The stored float32 vectors of each model, quantized afterwards (`ragbench/run_quant.py`). No model is run again.

| Variant | Bytes per passage | Index (MB) | Smaller by | nDCG@10 | Recall@10 | Recall@100 |
|---|---|---|---|---|---|---|
| e5-base float32 | 3072 | 8.84 | 1x | 80.3 | 98.6 | 100.0 |
| e5-base int8 x int8 | 768 | 2.21 | 4x | 75.3 | 97.4 | 99.9 |
| e5-base int8, float query | 768 | 2.21 | 4x | 80.3 | 98.6 | 100.0 |
| e5-base binary | 96 | 0.28 | 32x | 63.8 | 86.0 | 97.5 |
| e5-base binary + int8 rescoring | 96 | 0.28 | 32x | 78.2 | 94.4 | 97.5 |
| e5-small float32 | 1536 | 4.42 | 1x | 78.9 | 97.9 | 99.8 |
| e5-small int8 x int8 | 384 | 1.10 | 4x | 74.1 | 95.2 | 99.3 |
| e5-small int8, float query | 384 | 1.10 | 4x | 78.8 | 97.8 | 99.8 |
| e5-small binary | 48 | 0.14 | 32x | 40.6 | 60.1 | 83.5 |
| e5-small binary + int8 rescoring | 48 | 0.14 | 32x | 65.9 | 76.6 | 83.5 |
| minilm-paraphrase float32 | 1536 | 4.42 | 1x | 64.1 | 84.3 | 96.3 |
| minilm-paraphrase int8 x int8 | 384 | 1.10 | 4x | 63.3 | 83.9 | 96.3 |
| minilm-paraphrase int8, float query | 384 | 1.10 | 4x | 64.0 | 84.3 | 96.3 |
| minilm-paraphrase binary | 48 | 0.14 | 32x | 58.2 | 80.0 | 94.4 |
| minilm-paraphrase binary + int8 rescoring | 48 | 0.14 | 32x | 63.9 | 84.0 | 94.4 |

| Comparison (A - B) | Metric | Queries | Difference (points) | 95 % interval, clustered | 95 % interval, per query | p (clustered) | Resolved? |
|---|---|---|---|---|---|---|---|
| e5-base int8 x int8 - float32 | ndcg@10 | 1025 | -5.0 | -6.4 to -3.5 | -5.9 to -4.0 | 0.0001 | yes |
| e5-base int8 x int8 - float32 | recall@100 | 1025 | -0.1 | -0.3 to +0.0 | -0.3 to +0.0 | 1.0000 | no |
| e5-base int8, float query - float32 | ndcg@10 | 1025 | +0.0 | -0.0 to +0.1 | -0.0 to +0.1 | 0.5009 | no |
| e5-base int8, float query - float32 | recall@100 | 1025 | +0.0 | +0.0 to +0.0 | +0.0 to +0.0 | 1.0000 | no |
| e5-base binary - float32 | ndcg@10 | 1025 | -16.5 | -19.2 to -14.0 | -18.4 to -14.7 | 0.0001 | yes |
| e5-base binary - float32 | recall@100 | 1025 | -2.5 | -3.7 to -1.5 | -3.5 to -1.7 | 0.0001 | yes |
| e5-base binary + int8 rescoring - float32 | ndcg@10 | 1025 | -2.1 | -3.1 to -1.2 | -3.0 to -1.4 | 0.0001 | yes |
| e5-base binary + int8 rescoring - float32 | recall@100 | 1025 | -2.5 | -3.7 to -1.5 | -3.5 to -1.7 | 0.0001 | yes |
| e5-small int8 x int8 - float32 | ndcg@10 | 1025 | -4.7 | -6.4 to -3.1 | -5.8 to -3.7 | 0.0001 | yes |
| e5-small int8 x int8 - float32 | recall@100 | 1025 | -0.5 | -0.9 to -0.1 | -1.0 to -0.1 | 0.0609 | no |
| e5-small int8, float query - float32 | ndcg@10 | 1025 | -0.0 | -0.1 to +0.1 | -0.1 to +0.1 | 0.4944 | no |
| e5-small int8, float query - float32 | recall@100 | 1025 | +0.0 | +0.0 to +0.0 | +0.0 to +0.0 | 1.0000 | no |
| e5-small binary - float32 | ndcg@10 | 1025 | -38.2 | -41.4 to -35.0 | -40.6 to -35.8 | 0.0001 | yes |
| e5-small binary - float32 | recall@100 | 1025 | -16.3 | -19.3 to -13.5 | -18.6 to -14.0 | 0.0001 | yes |
| e5-small binary + int8 rescoring - float32 | ndcg@10 | 1025 | -13.0 | -15.4 to -10.6 | -15.0 to -10.9 | 0.0001 | yes |
| e5-small binary + int8 rescoring - float32 | recall@100 | 1025 | -16.3 | -19.3 to -13.5 | -18.6 to -14.0 | 0.0001 | yes |
| minilm-paraphrase int8 x int8 - float32 | ndcg@10 | 1025 | -0.8 | -1.8 to +0.3 | -1.5 to -0.0 | 0.1810 | no |
| minilm-paraphrase int8 x int8 - float32 | recall@100 | 1025 | +0.0 | -0.6 to +0.5 | -0.6 to +0.5 | 1.0000 | no |
| minilm-paraphrase int8, float query - float32 | ndcg@10 | 1025 | -0.1 | -0.2 to +0.1 | -0.2 to +0.1 | 0.2687 | no |
| minilm-paraphrase int8, float query - float32 | recall@100 | 1025 | +0.0 | +0.0 to +0.0 | +0.0 to +0.0 | 1.0000 | no |
| minilm-paraphrase binary - float32 | ndcg@10 | 1025 | -5.9 | -7.5 to -4.4 | -7.1 to -4.7 | 0.0001 | yes |
| minilm-paraphrase binary - float32 | recall@100 | 1025 | -1.9 | -2.8 to -1.0 | -2.8 to -1.0 | 0.0003 | yes |
| minilm-paraphrase binary + int8 rescoring - float32 | ndcg@10 | 1025 | -0.2 | -0.4 to +0.0 | -0.4 to +0.0 | 0.1444 | no |
| minilm-paraphrase binary + int8 rescoring - float32 | recall@100 | 1025 | -1.9 | -2.8 to -1.0 | -2.8 to -1.0 | 0.0003 | yes |

The five variants:

- **float32**: the vectors as the model returns them.
- **int8 x int8**: every dimension is mapped to 256 steps between its smallest and largest value over the corpus (`quantize_embeddings` of sentence-transformers, which clips). The query is mapped the same way, and the score is the dot product of the int8 numbers.
- **int8, float query**: the same 1-byte passages, turned back into floats (`dequantize_int8` in `ragbench/search.py`), compared with the query as the model returns it.
- **binary**: one bit per dimension (its sign), compared by Hamming distance.
- **binary + int8 rescoring**: the binary index finds 40 candidates, which are then reordered with the float query against their int8 vectors. The binary index sits in memory; the int8 vectors are read only for the 40 candidates.

What you gain, what it costs:

- **int8 storage is free if the query stays float.** The index becomes 4 times smaller and nDCG@10 does not move (e5-base 80.3 against 80.3; the difference is inside -0.0 to +0.1).
- **The same int8 index costs 5 points for the e5 models if the query is rounded too.** e5-base loses 5.0 points (interval -6.4 to -3.5), e5-small 4.7. Both rows use the same 1-byte passage codes, so the loss comes from how the score is computed: every dimension has its own step size and offset, and a dot product of the raw int8 numbers ignores them. How the score is computed matters more than the 8 bits. For minilm-paraphrase the same loss is 0.8 points and not resolved; why the models differ was not examined.
- **Binary alone costs a lot for the e5 models.** e5-base loses 16.5 points, e5-small 38.2. minilm-paraphrase, also with 384 dimensions, loses 5.9. So the loss depends on the model, not only on the number of bits.
- **Binary plus rescoring keeps most of the quality for e5-base.** It loses 2.1 points (interval -3.1 to -1.2) for an index 32 times smaller. Recall@100 still falls from 100.0 to 97.5: rescoring reorders only the first 40, so a passage the binary search ranks lower stays where it is. For e5-small the loss stays large (13.0 points).
- **On this corpus size none of this matters for memory.** 2876 passages take 8.8 MB in float32 with e5-base. It matters at millions of passages: per passage e5-base needs 3072 bytes in float32, 768 in int8 and 96 in binary (bytes per passage as in the table; no index of that size was built).
- **Search time on 2876 passages is 0.6 to 3.7 ms per query for every variant** (median, `search_one_query` in the result files) and says nothing about a large index.

### Reranking and the prompt-injection guard: not measured

The code is ready and was planned with two rerankers (bge-reranker-v2-m3, mmarco-mMiniLMv2) on the 20 best passages of the first stage, and one guard (protectai/deberta-v3-base-prompt-injection-v2) on deepset/prompt-injections with German and English examples counted separately.
None of them ran. On 25 November 2026 from about 12:50 until at least 13:30 (local time), the storage server that Hugging Face sends model files from (`us.aws.cdn.hf.co`) did not answer from this machine: every connection timed out, with Hugging Face's own transfer protocol (xet) and without it. The Hub's API itself answered, so the cards and revisions could be read and checked, but no weights could be downloaded after the first three models.
For the same reason granite-embedding-278m-multilingual and bge-m3 were not measured.
`python -m ragbench all` runs everything once the files can be downloaded. The rerank and guard code is covered only by the unit tests of its helper functions, not by a real run (see Limits).

## Licences

What each card says at the pinned revision (`ragbench/licences.py`). `python -m ragbench licences` fetched all 16 cards again on 25 November 2026 and found the same licence field and the same revision for every one (`results/licence-check.json`).

"Product use" is the verdict for a commercial product:

- **yes**: the licence allows commercial use and the card names nothing that restricts it.
- **check**: the licence allows commercial use, but the card itself says the model was trained on data whose own terms are non-commercial. Whether that binds the model is a legal question; it needs legal review before product use.
- **no**: the licence forbids commercial use. Such a model is out, however well it scores.

| Repository | Revision | Licence on the card | Product use | Measured | Note |
|---|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | `614241f622` | mit | check | yes | The card lists MS MARCO (500k pairs) in its supervised fine-tuning data. The MS MARCO datasets are intended for non-commercial research purposes only (https://microsoft.github.io/msmarco/, section Terms and Conditions, read 25 Nov 2026). |
| `intfloat/multilingual-e5-base` | `d128750597` | mit | check | yes | Same training data table as e5-small, with MS MARCO. |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `e8f8c21122` | apache-2.0 | yes | yes | The card does not name its training data. A paraphrase model, not trained for search. |
| `ibm-granite/granite-embedding-278m-multilingual` | `a9cb533849` | apache-2.0 | yes | planned; download failed | The card lists its training data and says it leaves MS MARCO out because of its licence. |
| `BAAI/bge-m3` | `5617a9f61b` | mit | yes | planned; download failed | The card links its fine-tuning data (Shitao/bge-m3-data); that data was not audited here. |
| `Alibaba-NLP/gte-multilingual-base` | `6dde5e2ebc` | apache-2.0 | yes | no | Not measured, for time. At this revision it no longer needs remote code (model_type gte). |
| `jinaai/jina-embeddings-v2-base-de` | `3f9eede875` | apache-2.0 | yes | no | Not measured: it loads Python code from another repository (jinaai/jina-bert-implementation) with trust_remote_code, which this benchmark does not run without a code review. |
| `jinaai/jina-embeddings-v3` | `ab036b023d` | cc-by-nc-4.0 | no | no | Non-commercial. Out for a product, also when run on own hardware. |
| `BAAI/bge-reranker-v2-m3` | `953dc6f6f8` | apache-2.0 | yes | planned; download failed | The card does not name its training data. |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `1427fd6529` | apache-2.0 | check | planned; download failed | Trained on mMARCO, a machine translation of MS MARCO. The MS MARCO datasets are intended for non-commercial research purposes only (https://microsoft.github.io/msmarco/, section Terms and Conditions, read 25 Nov 2026). |
| `jinaai/jina-reranker-v2-base-multilingual` | `9cfeff2df7` | cc-by-nc-4.0 | no | no | Non-commercial, and it needs trust_remote_code. Screened out before any compute was spent. |
| `protectai/deberta-v3-base-prompt-injection-v2` | `90c9989b1a` | apache-2.0 | yes | planned; download failed | English only by its own card, and the project is archived (no further updates). |
| `deepset/deberta-v3-base-injection` | `80dda00d0b` | mit | yes | no | Not measured: it was trained on deepset/prompt-injections, the evaluation data here. Its card reports accuracy 0.9914 on its evaluation set, which is 115 of 116, the size of the test split; so the test split was seen during training and gives no independent figure. |
| `meta-llama/Llama-Prompt-Guard-2-86M` | `a8ded8e697` | other (llama4) | check | no | Gated: card and weights need an account that accepts Meta's licence first. Not done here. |
| `deepset/germandpr` | `5129d02422` | cc-by-4.0 | yes | yes | Attribution required. The card's loader points to a zip on S3 that returns 404; the data comes from the Hub's Parquet conversion (refs/convert/parquet). |
| `deepset/prompt-injections` | `4f61ecb038` | apache-2.0 | yes | yes | The card states two licences: apache-2.0 at the top, cc-by-4.0 inside dataset_info. Both allow commercial use; CC BY 4.0 adds attribution. |

The findings that change a decision:

- **The two e5 models are MIT, but their own card lists MS MARCO in the fine-tuning data.** The MS MARCO terms say: "The MS MARCO datasets are intended for non-commercial research purposes only". The card licence alone would have said "go".
- **The mMARCO cross-encoder is Apache-2.0, and its card says it was trained on a machine translation of MS MARCO.** Same question.
- **granite-embedding-278m-multilingual says the opposite on its card:** it left MS MARCO out of its training data because of the licence. That makes it the model to measure next.
- **jina-embeddings-v3 and jina-reranker-v2-base-multilingual are CC BY-NC 4.0.** The jina-v3 card says this applies to use "on-premises within your company" too. They were screened out before any compute was spent.
- **protectai/deberta-v3-base-prompt-injection-v2 is English only by its own card, and the project is archived.** For German input that is a reason to look elsewhere before measuring.
- **deepset/deberta-v3-base-injection was trained on the evaluation data.** Its card reports accuracy 0.9914 on its evaluation set; 115 of 116 is 0.9914, and the test split of deepset/prompt-injections has 116 rows. A score on that split would not be an independent figure, so the model was not planned for this test.
- **Llama-Prompt-Guard-2-86M is gated** under Meta's Llama 4 licence; card and weights need an account that accepts it first.
- **GermanDPR's own loader is broken.** It points to a zip file on S3 that answers 404. Only the Hub's Parquet conversion is left, and that is what this benchmark pins and checks.
- **The deepset/prompt-injections card states two licences** (apache-2.0 at the top, cc-by-4.0 inside `dataset_info`). Both allow commercial use; CC BY 4.0 adds attribution.

## Entscheidungsgrundlage (decision summary)

Constraints assumed: commercial product, models on own CPU hardware, at most 500 ms per query at the 95th percentile for the whole retrieval step, German questions on German text.

**The pick from what was measured:** multilingual-e5-small, with int8 passage vectors and a float query.
It is not measurably worse than e5-base on this data (+1.4 points for e5-base, not resolved), it is 3.8 times faster to index and more than twice as fast per query (p95 29 ms, far below 500 ms), and int8 storage cost nothing measurable.
**Under one condition:** legal review accepts a model whose card names MS MARCO as training data. If it does not, both e5 models are out.

**What would change the pick:**

- **Legal says no to MS MARCO-trained models.** Then the measured field has only paraphrase-multilingual-MiniLM left, and it is 16 points worse; it should not be used for search. The next step would be to measure granite-embedding-278m-multilingual, whose card excludes MS MARCO, and bge-m3.
- **A reranker.** Recall@100 is near 100 %, so the gains are in the order of the top 10. A cross-encoder on the top 20 could change the ranking of first stages; this was not measured. Its latency on this CPU also decides whether it fits in 500 ms.
- **The real documents.** GermanDPR is Wikipedia. Supply-chain documents, tables, product codes and short German compound words may rank the models differently. The same harness runs on an own test set: corpus, questions and one relevant passage per question in the format of `data/germandpr-test/`.
- **A much larger corpus.** With millions of passages the index size and an approximate index matter, and binary with rescoring becomes interesting for e5-base (32 times smaller for 2.1 points here).

**What the data cannot tell:**

- Whether e5-base is better than e5-small. A difference of about 1.5 points in nDCG@10 is below what 1025 questions about 192 passages can resolve.
- Anything about granite, bge-m3, the rerankers or the guard.
- How the models rank on documents that are not Wikipedia, or on a corpus of realistic size.

## Method

### Data

The test split of [GermanDPR](https://huggingface.co/datasets/deepset/germandpr) by deepset (Möller, Risch and Pietsch, 2021), CC BY 4.0.
GermanDPR builds on GermanQuAD: annotators wrote German questions about German Wikipedia passages.
For every question the data has one positive passage (the one the question was written for) and three hard negatives, which deepset found with BM25 in German Wikipedia.

- **Corpus**: every distinct passage of the test split, 2876 passages (the same corpus as MTEB's GermanDPR task). Mean length 146 words.
- **Queries**: all 1025 questions. Each has exactly one relevant passage.
- **Reranking subset** (planned, not run): 300 questions drawn at random with seed 42 (`rerank_queries.txt`); the large reranker would use 100 of them, drawn from the 300 with the same seed.

Where the data comes from, exactly: the dataset card's own loader (`germandpr.py`, revision `5129d02422`) downloads a zip file from S3 that now answers `404 Not Found`.
The only copy left on the Hub is the Parquet conversion that Hugging Face made of it, on the branch `refs/convert/parquet`, commit `d0b867a49c`.
`python -m ragbench prepare` downloads `plain_text/test/0000.parquet` from that commit, checks its SHA-256, and writes the derived files to `data/germandpr-test/`.
`data/germandpr-test/manifest.json` records the source revision and the SHA-256 of every derived file. Every run checks the files against it first and stops if one has changed.
The derived files are in the repository, so the exact test data is versioned with the code. `.gitattributes` keeps Git from changing their line endings.

### Models

Every model is loaded at a fixed commit of its repository on Hugging Face (`ragbench/settings.py`).
Only models that load with the standard code of `transformers` and `sentence-transformers` are run: none needs `trust_remote_code`.

| Key | Repository | Params | Dims | Run? | Notes |
|---|---|---|---|---|---|
| e5-small | `intfloat/multilingual-e5-small` | 118 M | 384 | yes | prefixes `query: ` and `passage: ` as the card says |
| e5-base | `intfloat/multilingual-e5-base` | 278 M | 768 | yes | same prefixes |
| minilm-paraphrase | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118 M | 384 | yes | a paraphrase model, often the default in tutorials; its own limit of 128 tokens |
| granite-278m | `ibm-granite/granite-embedding-278m-multilingual` | 278 M | 768 | no, download failed | no prefix |
| bge-m3 | `BAAI/bge-m3` | 568 M | 1024 | no, download failed | dense vectors only |
| bge-reranker-v2-m3 | `BAAI/bge-reranker-v2-m3` | 568 M | | no, download failed | cross-encoder, planned on 100 queries |
| mmarco-minilm | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 118 M | | no, download failed | cross-encoder, planned on 300 queries |
| protectai-v2 | `protectai/deberta-v3-base-prompt-injection-v2` | 184 M | | no, download failed | prompt-injection classifier |

Models that were considered and left out on purpose are in the licence table, with the reason.

All embedding vectors are normalized; search is the dot product over all passages (exact, no approximate index).
Passages are cut at 512 tokens (128 for minilm-paraphrase).

### Metrics

Per query, then averaged over the queries (`ragbench/metrics.py`):

- **nDCG@10**: a hit at rank r is worth 1 / log2(r + 1); with one relevant passage this is 1 at rank 1, 0.63 at rank 2, 0.5 at rank 3, and 0 below rank 10.
- **Recall@10, @20, @100**: is the relevant passage among the first k? Recall@20 is the ceiling of a reranker that reorders the top 20.
- **MRR@10**: 1 / rank of the relevant passage, 0 below rank 10.

### Statistics

Two systems are always compared on the same queries, query by query (`ragbench/stats.py`, `ragbench/compare.py`).

- **Clustered paired bootstrap, 95 % interval.** GermanDPR has 1025 questions but only 192 distinct positive passages: several questions ask about the same passage, and a model that misreads a passage tends to miss all of them. These questions are not independent. The bootstrap therefore draws whole clusters (all questions with the same positive passage), 10,000 times, and takes the mean difference each time. The interval per query, which treats every question as independent, is shown next to it; it is narrower and too optimistic.
- **Paired sign-flip test (randomization test), clustered.** If both systems were equally good, the sign of each cluster's difference would be a coin toss. `p` is the share of 10,000 random sign patterns whose mean difference is at least as far from 0 as the observed one.
- **"Resolved"** means the clustered 95 % interval does not contain 0. Where it does, the data cannot say which system is better.
- Many comparisons are made, and no correction for multiple testing is applied. A single `p` near 0.05 among many should be read with that in mind.

The unit tests check every metric and both statistics on hand-made cases (`tests/`).

### Cost and speed

Measured on one laptop CPU, one model at a time, with no other heavy work running (`results/*/*.json`, field `machine`): AMD Ryzen 7 5800HS (8 cores, 16 threads), 15.4 GB RAM, Windows 11, Python 3.12, PyTorch 2.15 for CPU with 8 threads, no GPU.
Every model runs in its own process, so that the peak memory is its own.

- **Corpus encode**: wall time to embed all 2876 passages, batch size 32.
- **One query p50 / p95**: 200 queries, one at a time (batch size 1), after 5 warm-up queries: embed the query and search all passages. This is what a live system sees.
- **Rerank p50 / p95** (not run): per query, scoring its 20 candidate passages in one batch.
- **Peak RAM**: the peak working set of the process (Windows), including the Python runtime and PyTorch.
- **Index**: number of passages times dimensions times bytes per value.
- **Weights**: size of the weight file on disk.

## Limits

- **One dataset, Wikipedia only.** GermanDPR's passages are German Wikipedia paragraphs. Company documents may rank the models differently.
- **A small, easy corpus.** 2876 passages, and the three hard negatives of each question were found by BM25. Recall@100 is near 100 %. With a corpus of millions of passages all figures would be lower and the gaps could be larger.
- **One relevant passage per question, and the negatives were not checked by people.** A hard negative that also answers the question counts as wrong. This lowers all models alike, but not necessarily by the same amount.
- **The questions were written while looking at the passage.** They share many words with it. Real users' questions do not.
- **The models may have seen the data.** GermanDPR and German Wikipedia are public since 2021 and earlier. No card lists GermanDPR as training data, but none of this was checked. The point of the run is the comparison under the same conditions.
- **Questions are clustered.** 1025 questions ask about 192 distinct passages. The clustered statistics take this into account; they are wider, and that is why e5-base against e5-small is not resolved.
- **No correction for multiple testing.** There are 28 comparisons in `results/comparisons.json`. The conclusions above rest on differences with p = 0.0001 or on intervals far from 0, not on borderline cases.
- **Speed is one run on one laptop CPU**, AMD Ryzen 7 5800HS, whose power settings were not controlled. Other machines will differ.
- **The rerank and guard code has not run end to end.** Its helper functions are unit-tested (`rerank_one`, `wilson_interval`, `guess_language`), and the `CrossEncoder` calls were checked against the installed sentence-transformers, but no reranker and no classifier was loaded. The first real run may still find a bug.
- **CI has not run yet.** GitHub did not answer from this machine on the day, so the repository was not pushed and the workflow is untested.
- **The language split of the guard data would be a guess.** deepset/prompt-injections has no language column; `ragbench/language.py` counts German and English function words. It is tested on four sentences, not validated.

## How to reproduce

Python 3.12 on any system with about 6 GB free disk:

```
python -m venv .venv
.venv/Scripts/activate            # Linux: source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
python -m ragbench prepare        # optional: rebuilds data/ and must give the same SHA-256 sums
python -m ragbench all            # every model, comparisons, tables
python -m ragbench licences       # fetches every card again and checks the licence field
pytest
ruff check . && ruff format --check .
```

A new run overwrites `results/`. The tables in this README are the output of `python -m ragbench report`.
CI (`.github/workflows/ci.yml`) runs lint, the unit tests, the manifest check and `python -m ragbench smoke`: 8 queries against 206 passages with e5-small, then mmarco-minilm and the guard on 8 texts. It does not run the benchmark.
The embedding and quantization parts of the smoke run were run on this machine; the rerank and guard parts were not, because their models could not be downloaded. The CI workflow has not run yet (see Limits).

## Files

| File | What it does |
|---|---|
| `ragbench/settings.py` | Every fixed choice: data revision and checksum, models and revisions, seeds, depths, sizes. |
| `ragbench/data.py` | Builds corpus, queries and relevance labels from GermanDPR; writes and checks the manifest. |
| `ragbench/fetch.py` | Downloads data and models at a fixed revision; SHA-256 of a file. |
| `ragbench/metrics.py` | nDCG, recall and MRR for one query. |
| `ragbench/stats.py` | Clustered paired bootstrap, paired sign-flip test, Wilson interval. |
| `ragbench/search.py` | Exact search over float32, int8 and binary vectors; binary search with int8 rescoring. |
| `ragbench/run_embed.py` | One embedding model: encode, search, score, time, memory. |
| `ragbench/run_rerank.py` | One reranker on the first stage's top 20. |
| `ragbench/run_quant.py` | float32, int8 and binary versions of the stored vectors. |
| `ragbench/run_guard.py` | The prompt-injection classifier, one text at a time. |
| `ragbench/language.py` | A small word-list guess: German or English. |
| `ragbench/licences.py` | The licence table and the script that checks it against the cards. |
| `ragbench/compare.py` | All paired comparisons, written to `results/comparisons.json`. |
| `ragbench/report.py` | Prints the README tables from the result files. |
| `ragbench/__main__.py` | The command line. |
| `data/germandpr-test/` | The derived test data and its manifest. |
| `results/` | One JSON file per run, with per-query values; `comparisons.json`; `licence-check.json`; `pip-freeze.txt`. |
| `tests/` | Unit tests for metrics, statistics, search, data building and the language guess. |

## Sources and licences

- Code in this repository: MIT, see `LICENSE`.
- GermanDPR: Möller, Risch, Pietsch, *GermanQuAD and GermanDPR: Improving Non-English Question Answering and Passage Retrieval*, 2021, <https://arxiv.org/abs/2104.12741>. CC BY 4.0. The files in `data/germandpr-test/` are derived from its test split: passages deduplicated and given ids, questions numbered, one relevance label per question. Nothing else is changed.
- deepset/prompt-injections: card states Apache-2.0 and CC BY 4.0 (see the licence table). This repository contains none of its texts; the result file holds row ids, labels and decisions only.
- The models: see the licence table. This repository contains no model weights.
