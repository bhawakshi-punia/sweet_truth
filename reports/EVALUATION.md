# How performance is judged

Each ingredient is a **sweetener** (positive) or **other** (negative). A model's probability of 0.5 or more counts as "sweetener".

| | Predicted sweetener | Predicted other | |
| --- | --- | --- | --- |
| **Actual sweetener** | True positive (TP) | False negative (FN), *Type II error* | **Sensitivity** (recall) = TP / (TP + FN) |
| **Actual other** | False positive (FP), *Type I error* | True negative (TN) | **Specificity** = TN / (TN + FP) |
| | **Precision** = TP / (TP + FP) | **Negative predictive value** = TN / (TN + FN) | **Accuracy** = (TP + TN) / all |

F1 = 2 × precision × sensitivity / (precision + sensitivity).

## What matters most

- **Sensitivity comes first.** A missed sweetener (false negative) is the costly error for someone avoiding sugar.
- **Precision keeps false alarms in check.** A harmless ingredient flagged is a nuisance, and the reader can check it.
- **Accuracy alone misleads.** Only about 13% of ingredients are sweeteners, so "never a sweetener" would already be 87% accurate.

## The sets

| Set | Tests | Note |
| --- | --- | --- |
| Train, validation, test | Learning, and lists not trained on | Every test ingredient also appears in training: this mostly measures memory |
| **Unseen ingredients** | Names never seen in training (5-fold, grouped by ingredient) | The main test |
| Challenge | 46 real names, including look-alikes | Small: one name moves accuracy by about 2 points |
| Photos | End to end, from a photo | Also depends on the text recognition |

v1 retrained exactly as in the original notebook scores 0.98 on its own list-level measure (the notebook reported 0.97; one list differs, most likely due to a newer lemmatiser). So the split and measure match. Details on how the sets are built: [DATASET.md](../src/sweet_truth/training/DATASET.md).
