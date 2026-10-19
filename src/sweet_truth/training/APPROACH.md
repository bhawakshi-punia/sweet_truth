# Approach: model v1 and model v2

## Side by side

| | v1 (original) | v2 (current app model) |
| --- | --- | --- |
| Trained on | Whole lists: "does this list contain a sweetener?" | Single ingredients: "is this a sweetener?", which is how the app uses it |
| Input cleaning | Bracketed text and full stops deleted, lower-cased, lemmatised with spaCy, stop words dropped | Spaces tidied and edge punctuation trimmed; brackets and their contents (e.g. "(955)") kept |
| Splitting a list | Commas, colons, "and", "or" (after brackets are deleted) | The same, but ignoring separators inside brackets; a new word right after a closing bracket also starts a new ingredient (photos often lose that comma) |
| Features | TF-IDF on whole words | TF-IDF on letter chunks of 1 to 6 characters within words (`char_wb`, sublinear term frequency) |
| Classifier | Random forest: 451 trees, min_samples_split 6 (tuned in the notebook) | Logistic regression: C = 10, balanced class weights |
| Size, speed | 7.2 MB, about 5.5 ms per ingredient | 0.2 MB, about 0.02 ms per ingredient |
| Explains a prediction | No | Yes: the parts of the ingredient that pushed it towards "sweetener" |

**Why letter chunks matter:** v1 only knows whole words it has seen. Letter chunks let v2 recognise word parts that many sweeteners share, such as *-ose* (fructose, isomaltulose), *-itol* (sorbitol, maltitol), *syrup* and *sugar*. That lets it handle names it has never seen.

## Which change matters: features or classifier?

v1 and v2 differ in both the features and the classifier, so all four combinations were trained the same way (on single ingredients) and scored on unseen ingredients:

| Features | Classifier | Unseen recall | Unseen F1 | Challenge accuracy |
| --- | --- | --- | --- | --- |
| words | random forest (v1's) | 0.38 | 0.54 | 0.74 |
| words | logistic regression | 0.47 | 0.62 | 0.74 |
| letter chunks | random forest | 0.55 | 0.69 | 0.83 |
| **letter chunks** | **logistic regression (v2)** | **0.87** | **0.89** | 0.78 |

- **Both changes help, letter chunks the most.** Logistic regression adds a large further gain on top of letter chunks: it weighs thousands of sparse chunk features together, where trees split on a few.
- **The challenge set disagrees slightly.** Letter chunks + random forest scores higher there (38 vs 36 of 46 correct). With only 46 names, two names make that difference. The unseen folds (391 names) were used to choose, as decided beforehand.
- **Tuning note:** v2's chunk lengths (1 to 6) and C = 10 were the best of 96 settings tried on the unseen folds. Because they were chosen on the same folds they are scored on, v2's unseen score is slightly optimistic. With the untuned 2-to-5 setting it found 85% of unseen sweeteners instead of 87%.

## How the deployed model is fitted

The numbers in the reports come from models fitted on the 800 training lists only. The model the app uses (`models/sweet_truth_v2.joblib`) is then refitted on the ingredients of all 1,000 lists with the same settings (`sweet-truth model export`), as v1's deployed pickle was.

## How the "why" is computed

For a flagged ingredient, each letter chunk's push towards "sweetener" is its TF-IDF value times the logistic regression's weight for that chunk. The strongest chunks are located in the ingredient and overlapping ones merged, so *ugar* and *suga* read as **sugar**. A part only a letter or two short of its word is shown as the whole word (**sorbitol** rather than *-orbitol*). A "-" marks a cut inside a word: **-ose** is a word ending, **isomalt-** a word start. The app shows up to three, e.g. "because of: isomalt-, -ose". See `explain()` in [`../text.py`](../text.py).

## The transformer experiment (ELECTRA-small, RoBERTa-base)

- **ELECTRA-small:** a spaCy text classifier on ELECTRA-small (about 14M parameters) was trained in the same way (`configs/textcat_transformer.cfg`). It reached a perfect test score but found only 70% of unseen sweeteners, against 87% for v2. With only about 360 distinct names to learn from, it memorised them: its scores are almost always exactly 0 or 1. It is also 55 MB and needs PyTorch, which does not fit the free hosting.
- **RoBERTa-base:** training was stopped part-way because it overheated the laptop. [`notebooks/model_v2_colab_train.ipynb`](../../../notebooks/model_v2_colab_train.ipynb) can train it on a Colab GPU if wanted.

## Known limits

- **Sweeteners with no telltale spelling are missed by every model:** thaumatin, sodium cyclamate, neohesperidin dihydrochalcone, rebaudioside A, treacle, monk fruit extract. A curated list of sweeteners and their INS / E-numbers (950–969) is the planned fix.
- **Look-alikes are flagged:** ethyl maltol (a flavour) and glucose oxidase (an enzyme), because they share parts with maltose and glucose.
- **Biggest lever for any model:** more labelled ingredient names.
