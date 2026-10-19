# Approach

## v1 vs v2

| | v1 (original) | v2 (in the app) |
| --- | --- | --- |
| Trained on | Whole lists ("contains a sweetener?") | Single ingredients, as the app uses it |
| Input | Brackets deleted, lemmatised words | Brackets kept ("(955)"), text as written |
| Features | TF-IDF on whole words | TF-IDF on letter chunks of 1 to 6 characters (`char_wb`) |
| Classifier | Random forest (451 trees) | Logistic regression (C = 10, balanced) |
| Size | 7.2 MB | 0.2 MB |
| Explains itself | No | Yes |

**Why letter chunks:** v1 knows only whole words it has seen. Letter chunks catch parts many sweeteners share, such as *-ose*, *-itol*, *syrup* and *sugar*, so v2 can recognise new names.

**Splitting a list:** at commas, colons, "and" and "or", ignoring separators inside brackets. A new word right after a closing bracket also starts a new ingredient, because photos often lose that comma.

## Features or classifier?

Unseen-ingredient F1 when all four combinations are trained the same way: words + forest 0.54, words + logistic 0.62, chunks + forest 0.69, **chunks + logistic 0.89**.

- **Both changes help; the chunks matter most.** v2 was chosen on these unseen folds, as decided beforehand.
- **The challenge set disagreed slightly:** chunks + forest got 38 of 46 right, v2 36.
- **Tuning caveat:** v2's settings were the best of 96 tried on the same folds, so its unseen score is slightly optimistic. Untuned, it found 85% of unseen sweeteners rather than 87%.

## The deployed model

The reported numbers come from models fitted on the 800 training lists. The app's model (`models/sweet_truth_v2.joblib`) is refitted on all 1,000 lists with the same settings (`sweet-truth model export`), as v1's was.

## The "why"

For a flagged ingredient, each letter chunk's push towards "sweetener" is its TF-IDF value times the model's weight for it. The strongest chunks are located in the ingredient and merged, so *ugar* + *suga* reads **sugar**. A part a letter or two short of its word becomes the whole word. A "-" marks a cut inside a word: **-ose**, **isomalt-**. See `explain()` in `src/sweet_truth/text.py`.

## Transformer experiment

- **ELECTRA-small:** a spaCy text classifier on ELECTRA-small (14M parameters) scored perfectly on the test set but found only 70% of unseen sweeteners. With about 360 distinct names to learn from, it memorised them. It also needs PyTorch, which is too big for the free hosting.
- **RoBERTa-base:** stopped part-way, since it overheated the laptop. The Colab notebook can train it.
