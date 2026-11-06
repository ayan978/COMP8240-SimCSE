# COMP8240 — Replicating and Extending SimCSE

This repository contains the code, data-processing workflow, experimental
results, reproducibility evidence and report materials for the COMP8240
project:

**Replicating and Extending SimCSE**

The project is based on:

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021.
*SimCSE: Simple Contrastive Learning of Sentence Embeddings*.
EMNLP 2021.

Official implementation:
https://github.com/princeton-nlp/SimCSE

The official repository was pinned to commit:

49fa580a853752ede55b8c76d9debf748e214d3f

## Project scope

The project contains three main experimental components:

1. reproduction of the original SimCSE STS evaluation and fresh supervised
   and unsupervised BERT-base training;
2. evaluation of the released supervised SimCSE checkpoint on an external
   sentence-similarity dataset from Zhao and Hu;
3. construction and evaluation of a new 400-pair controlled semantic
   diagnostic dataset, including Gemini LLM-as-judge evaluation and
   comparison with an earlier manually scored 80-pair pilot.

## Main results

### Original SimCSE fresh-training reproduction

| Setting | Paper Avg. STS | Reproduction | Difference |
|---|---:|---:|---:|
| Supervised BERT-base | 81.57 | 81.20 | -0.37 |
| Unsupervised BERT-base | 76.25 | 74.84 | -1.41 |

Evaluation of the official released supervised checkpoint also reproduced
the official repository reference result at approximately 81.58 average STS.

### New-data evaluation

| Evaluation | N | Spearman | Pearson |
|---|---:|---:|---:|
| Zhao and Hu cleaned dev set | 2,998 | 0.6008 | 0.6759 |
| Constructed controlled dataset | 400 | 0.3924 | 0.4900 |
| Gemini vs manual pilot scores | 80 | 0.8688 | 0.9163 |

The controlled dataset shows especially strong SimCSE overestimation on
causal-direction reversal, high lexical-overlap role reversal, negation and
antonym substitution.

## Repository structure

- `replication/` — SimCSE retraining scripts, environment details and run evidence
- `data_processing/` — dataset preparation and construction scripts
- `data/constructed/` — project-created controlled datasets
- `data/zhao_hu/` — location for local third-party Zhao and Hu data
- `evaluation/` — Zhao and Hu, constructed-data and Gemini evaluation scripts
- `analysis/` — result summarisation and report-table preparation
- `results/` — archived experimental outputs
- `figures/` — report-ready figures
- `report/` — ACL LaTeX report source

## Environments

Two software environments are intentionally documented separately.

### Extension and new-data evaluation

Verified local environment:

- Python 3.12.7
- pandas 2.2.2
- SciPy 1.13.1
- PyTorch 2.15.0
- Transformers 5.22.1
- Pydantic 2.14.3
- google-genai 3.0.0

Install the direct dependencies with:

    python -m pip install -r requirements.txt

### Fresh SimCSE retraining

The fresh reproduction used an older software stack appropriate to the
official 2021 SimCSE implementation.

Full details are in:

    replication/REPRODUCTION_ENVIRONMENT.md

The original completed-run logs, configs, trainer states and environment
records are retained under:

    replication/training/comp8240_supervised_evidence/
    replication/training/comp8240_unsupervised_evidence/

## Reproducing the original SimCSE training experiments

Clone the official SimCSE repository and check out the pinned commit:

    git clone https://github.com/princeton-nlp/SimCSE.git
    cd SimCSE
    git checkout 49fa580a853752ede55b8c76d9debf748e214d3f

Prepare the official SimCSE training data and SentEval STS data according to
the authors' repository instructions.

Then, from this COMP8240 repository, set:

    export SIMCSE_DIR="$HOME/SimCSE"
    export PYTHON_BIN="/path/to/reproduction/python"

Run:

    bash replication/run_supervised_reproduction.sh
    bash replication/run_unsupervised_reproduction.sh

The historical Vast.ai run scripts are also retained inside the two evidence
directories. Their `/workspace/SimCSE` paths are preserved intentionally
because they record the filesystem used during the completed GPU runs.

## Zhao and Hu external dataset

The third-party dataset is not redistributed in this repository.

Source paper:

Zhao and Hu (2026), PLOS ONE:
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0345540

Place the downloaded source CSV files at:

    data/zhao_hu/local/cw2_train.csv
    data/zhao_hu/local/cw2_dev.csv

Prepare the exact cleaned files used in the project:

    python data_processing/prepare_zhao_hu.py

The verified preprocessing is:

1. remove the exported `Unnamed: 0` CSV index column;
2. verify that there are no missing values;
3. remove exact duplicate rows.

Verified counts:

| Split | Raw | Duplicates removed | Clean |
|---|---:|---:|---:|
| Train | 11,498 | 40 | 11,458 |
| Dev | 3,000 | 2 | 2,998 |

Evaluate the cleaned dev set with the released supervised SimCSE checkpoint:

    python evaluation/evaluate_zhao_hu.py

Expected result:

    Spearman: 0.6008
    Pearson:  0.6759

## Constructed controlled dataset

### Pilot

The initial project pilot contains 80 manually scored sentence pairs:
10 pairs in each of eight semantic categories.

Pilot files are retained for provenance:

    data_processing/build_constructed_pilot.py
    evaluation/evaluate_constructed_pilot.py
    analysis/summarize_constructed_results.py

These are pilot-stage files rather than the final experimental pipeline.

### Final 400-pair dataset

The final controlled dataset contains:

- 400 sentence pairs
- 8 semantic categories
- 50 pairs per category
- 100 meaning-preserved pairs
- 300 meaning-changed pairs
- 80 retained pilot pairs
- 320 additional controlled pairs
- 0 missing sentence values
- 0 exact or reversed duplicate pairs

Generate and validate it with:

    python data_processing/build_constructed_final.py

The resulting dataset is:

    data/constructed/constructed_final_400.csv

## Gemini LLM-as-judge

The final judge model is:

    gemini-3.5-flash-lite

The judge is not shown the dataset category or expected-relation label.

Evaluation settings include:

- temperature = 0
- similarity scale = 0.0 to 5.0
- half-point increments
- batch size = 10
- shuffle seed = 8240
- structured JSON output
- automatic retry and checkpoint/resume

Set the API key locally:

    export GEMINI_API_KEY="PASTE_YOUR_GEMINI_API_KEY_HERE"

Do not commit the API key.

Run:

    python evaluation/judge_constructed_gemini_35_flash_lite.py

The exact judge configuration, final judgments and raw API outputs are
archived under:

    results/constructed/

Agreement between Gemini scores and the manually scored 80-pair pilot:

    Spearman: 0.8688
    Pearson:  0.9163

This is reported as manual–LLM agreement, not inter-annotator agreement
between multiple human annotators.

## Final constructed-data SimCSE evaluation

Run:

    python evaluation/evaluate_constructed_final.py

Expected overall result:

    N:        400
    Spearman: 0.3924
    Pearson:  0.4900

Mean scores by expected semantic relation:

Meaning-preserved pairs:

    Gemini mean:        4.905 / 5
    SimCSE cosine mean: 0.9529

Meaning-changed pairs:

    Gemini mean:        1.643 / 5
    SimCSE cosine mean: 0.8017

Category summaries and individual diagnostic failures are stored under:

    results/constructed/

## Final report summaries

Generate consolidated report-ready numerical summaries with:

    python analysis/build_final_report_summary.py

Outputs are written to:

    results/final_report/

## Primary final-project scripts

The principal files supporting the final report are:

    data_processing/prepare_zhao_hu.py
    data_processing/build_constructed_final.py
    evaluation/evaluate_zhao_hu.py
    evaluation/judge_constructed_gemini_35_flash_lite.py
    evaluation/evaluate_constructed_final.py
    analysis/build_final_report_summary.py
    replication/run_supervised_reproduction.sh
    replication/run_unsupervised_reproduction.sh

Files containing `pilot` in their names are retained to document the earlier
experimental stage and are not the primary final pipeline.

## Reproducibility and data notes

- Final project scripts use repository-relative paths.
- The official SimCSE source commit is pinned.
- Fresh-training environment details are archived.
- Original training logs and trainer states are retained.
- Gemini credentials are read only from the environment.
- The exact Gemini model string, prompt/configuration and raw outputs are archived.
- The project-created 400-pair dataset is included in the repository.
- Third-party Zhao and Hu data are not redistributed.
- Zhao and Hu preprocessing is completely reproducible from the source CSVs.
