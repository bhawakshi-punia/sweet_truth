# How performance is judged

Each model decides, ingredient by ingredient, whether it is a **sweetener** (the positive class) or **other**. A probability of 0.5 or more counts as "sweetener".

## The confusion matrix

| | Predicted sweetener | Predicted other | |
| --- | --- | --- | --- |
| **Actually a sweetener** | True positive (TP) | False negative (FN), *Type II error* | **Sensitivity** (recall) = TP / (TP + FN) |
| **Actually other** | False positive (FP), *Type I error* | True negative (TN) | **Specificity** = TN / (TN + FP) |
| | **Precision** = TP / (TP + FP) | **Negative predictive value** = TN / (TN + FN) | **Accuracy** = (TP + TN) / all |

**F1** = 2 × precision × sensitivity / (precision + sensitivity) combines the two measures about sweeteners into one number.

In the figures ([`figures/`](figures/)), correct cells are violet and errors amber, and every cell is also labelled in words.

## Which numbers matter most here

- **Sensitivity (recall) comes first.** A **false negative**, a sweetener the app fails to flag, is the costly mistake: someone avoiding sugar would trust the label.
- **Precision is the guard against false alarms.** A **false positive**, flagging something harmless, is a nuisance, but the person can check that ingredient.
- **Accuracy alone misleads.** Only about 13% of ingredients are sweeteners, so a model that never says "sweetener" would still be 87% accurate. That is why the figures show the full matrix and the summary leads with precision, recall and F1.

## What each set tests

| Set | Question it answers | Caveat |
| --- | --- | --- |
| Train | Did the model learn the training data? | Says nothing about new data |
| Validation, Test | Does it work on lists it was not trained on? | Every test ingredient also appears in training, so this mostly measures remembering names; every model scores near 1.0 |
| **Unseen ingredients** | Can it recognise ingredients it has never seen? | The main test of whether v2's letter chunks help |
| Challenge | The same, on 46 real-world names chosen by hand, including look-alikes | Small: one name moves accuracy by about 2 points |
| Photos | End to end, from a photo: are the printed sweeteners found? | Only 8 photos; also depends on the text recognition |

Also reported: **list-level accuracy** on the test lists, where a list counts as flagged if any of its ingredients is. That is the original notebook's 97% measure. v1 retrained exactly as in the notebook scores 0.98 on it (one list differs, likely due to a newer lemmatiser), which confirms that the split and measure match.

See [`../../src/sweet_truth/training/DATASET.md`](../../src/sweet_truth/training/DATASET.md) for how the sets are built.
