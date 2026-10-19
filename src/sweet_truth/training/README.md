# Training and evaluation

Code to train model v2 (letter chunks + logistic regression, the model the app uses), compare it with v1, and run the transformer experiment.

- [DATASET.md](DATASET.md): the data, how lists become labelled ingredients, and every evaluation set.
- [APPROACH.md](APPROACH.md): v1 vs v2, the features-vs-classifier comparison, how the deployed model is fitted and how the "why" works, and the transformer experiment.
- Results: [`reports/model_v2/`](../../../reports/model_v2/README.md); how they are judged: [EVALUATION.md](../../../reports/model_v2/EVALUATION.md).

## Setup

```bash
uv sync --all-extras   # the `train` extra adds spacy-transformers, PyTorch, pandas and matplotlib
```

## Commands

```bash
uv run sweet-truth model evaluate     # compare v1, v2 (and any trained transformer); writes reports/model_v2/
uv run sweet-truth model export       # refit v2 on all 1,000 lists -> models/sweet_truth_v2.joblib (the app's model)
```

v1 and v2 train in seconds inside `evaluate` and `export`. Only the transformer experiment needs separate training:

```bash
uv run sweet-truth model prepare                                     # write spaCy corpora
uv run sweet-truth model train --transformer electra-small           # main model (~15 min on a Mac CPU)
uv run sweet-truth model train --transformer electra-small --folds   # one per unseen-ingredient fold (~45 min)
```

- **Other transformers:** `--transformer roberta-base`, or any Hugging Face name.
- **GPU:** `--gpu 0` uses one.
- **Colab:** RoBERTa-base is heavy for a laptop; [`notebooks/model_v2_colab_train.ipynb`](../../../notebooks/model_v2_colab_train.ipynb) trains it on a Colab GPU with the same `train_model()` function.

## Files

| File | What it does |
| --- | --- |
| `data.py` | Loads the CSVs, splits lists back into labelled ingredients, reproduces the notebook's split, builds the unseen-ingredient folds, loads the challenge set and photo expectations |
| `baselines.py` | v1 retrained as in the notebook and used as in the app; the four feature × classifier combinations; `fit_v2()` |
| `evaluate.py` | Scores every model on every set (confusion-matrix counts, sensitivity, specificity, precision, NPV, accuracy, F1), runs the comparison, writes the reports |
| `figures.py` | The confusion-matrix figures in `reports/model_v2/figures/` |
| `train.py` | `export_v2()` for the app's model; `train_model()` / `train_folds()` for the transformer experiment (spaCy's trainer, in-process) |
| `corpus.py`, `configs/textcat_transformer.cfg` | spaCy corpora and config for the transformer experiment |
| `cli.py` | The `sweet-truth model` commands |

The app-side code lives outside this folder:
- `../classifiers.py`: a shared interface for comparing models;
- `../text.py`: `split_ingredients`, `clean_ingredient` and `explain`, used by the app.

Trained transformer folders (`models/v2/`) and corpora (`corpus/`) are not committed; `models/sweet_truth_v2.joblib` and the reports are.
