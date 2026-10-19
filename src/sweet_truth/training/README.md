# Training and evaluation

Trains model v2 (the app's model), compares it with v1, and holds the transformer experiment.

- [DATASET.md](DATASET.md): the data and every evaluation set
- [APPROACH.md](APPROACH.md): v1 vs v2, the deployed model, the "why", and the transformer experiment
- Results: [`reports/`](../../../reports/README.md)

## Commands

```bash
uv sync --all-extras                  # the `train` extra adds spacy-transformers, PyTorch, pandas, matplotlib
uv run sweet-truth model evaluate     # compare all models; writes reports/
uv run sweet-truth model export       # refit v2 on all 1,000 lists -> models/sweet_truth_v2.joblib
```

v1 and v2 train in seconds inside those commands. The transformer experiment trains separately:

```bash
uv run sweet-truth model prepare                                     # spaCy corpora
uv run sweet-truth model train --transformer electra-small [--folds] # ~15 min (+45 for folds) on a Mac CPU
```

`--transformer roberta-base` (or any Hugging Face name) and `--gpu 0` also work. For a laptop, RoBERTa-base is better trained on Colab with [`notebooks/colab_transformer_training.ipynb`](../../../notebooks/colab_transformer_training.ipynb).

## Files

| File | Role |
| --- | --- |
| `data.py` | Load the CSVs, split lists into labelled ingredients, the notebook's split, unseen folds, challenge set |
| `baselines.py` | v1 as in the notebook and the app; the feature × classifier combinations; `fit_v2()` |
| `evaluate.py`, `figures.py` | Metrics on every set, the comparison, JSON reports and confusion-matrix figures |
| `train.py` | `export_v2()`; `train_model()` / `train_folds()` for the transformer (spaCy, in-process) |
| `corpus.py`, `configs/` | Transformer corpora and spaCy config |
| `cli.py` | The `sweet-truth model` commands |
