# SimCSE Fresh-Training Reproduction Environment

The fresh supervised and unsupervised SimCSE replications used the official Princeton NLP SimCSE repository at commit:

49fa580a853752ede55b8c76d9debf748e214d3f

## Verified software environment

- Python 3.8.20
- PyTorch 1.7.1+cu110
- Transformers 4.2.1
- datasets 1.18.4
- NumPy 1.23.5
- PyArrow 10.0.1
- SciPy 1.10.1
- pandas 1.5.3
- scikit-learn 1.2.2
- CUDA enabled

The completed runs used an NVIDIA A100 SXM4 80 GB GPU.

## Training data

Supervised:

- file: data/nli_for_simcse.csv
- examples: 275,601 excluding the CSV header
- SHA-256: 0747687ec3594fa449d2004fd3757a56c24bf5f7428976fb5b67176775a68d48

Unsupervised:

- file: data/wiki1m_for_simcse.txt
- sentences: 1,000,000
- SHA-256: 7b1825863a99aa76479b0456f7c210539dfaeeb69598b41fb4de4f524dd5a706

The authors' released processed training files were used unchanged.

## Archived evidence

The exact completed-run evidence is retained in:

- replication/training/comp8240_supervised_evidence/
- replication/training/comp8240_unsupervised_evidence/

These directories contain the original run scripts, environment records, configurations, trainer states, training logs and final STS test outputs.

The archived run scripts contain /workspace/SimCSE because that was the actual filesystem path on the Vast.ai training instance. They are deliberately preserved unchanged as provenance records.

## Portable scripts

Machine-independent versions are provided as:

- replication/run_supervised_reproduction.sh
- replication/run_unsupervised_reproduction.sh

Set SIMCSE_DIR to the official SimCSE clone and PYTHON_BIN to the Python executable from the reproduction environment before running them.

## Final fresh-training results

Supervised SimCSE-BERT-base:

- paper average STS Spearman: 81.57
- reproduced average: 81.20
- difference: -0.37

Unsupervised SimCSE-BERT-base:

- paper average STS Spearman: 76.25
- reproduced average: 74.84
- difference: -1.41
