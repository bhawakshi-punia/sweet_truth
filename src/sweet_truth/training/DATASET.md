# Dataset

v1 and v2 use the same data; the original CSVs are unchanged.

| File | Rows | Content |
| --- | --- | --- |
| `data/final_ingredient_dataset.csv` | 392 | Single ingredients labelled sweetener (1) or not (0); 53 sweeteners |
| `data/ingredients_sweeteners_list.csv` | 1,000 | Synthetic lists of 3 to 7 rows from the file above, with the sweeteners each contains; 491 contain one |

## From lists to labelled ingredients

Every list splits back exactly into the rows it was built from (`data.py`, `decompose`). That gives each ingredient occurrence an exact label: 13% of them are sweeteners. v2 trains on these single ingredients.

## Quirks of the data (left as is)

- **Spelling variants:** "Maltitol" and "MALTITOL (965)" are the same ingredient, so 392 rows are 360 distinct ingredients.
- **One label conflict:** "Maltodextrin" is labelled 1 but "MALTODEXTRIN" 0. This explains the few training errors.
- **Multi-ingredient rows:** some rows hold several ingredients, e.g. "fruit and vegetable juices".
- **One duplicated row:** "CHOCO CREAM" appears twice.

## Evaluation sets

| Set | Built from | Size |
| --- | --- | --- |
| Train / validation / test | The original notebook's split: `train_test_split(random_state=101)` 80 / 20, then the 20% halved | 800 / 100 / 100 lists (4,035 / 512 / 506 ingredients) |
| Unseen ingredients | 5-fold cross-validation over the distinct rows, grouped so all spellings of an ingredient share a fold | 391 |
| Challenge | `data/challenge_ingredients.csv`: real names not in the data, labelled by hand (24 sweeteners, 22 not) | 46 |
| Photos | `data/photo_sweeteners.csv`: sweeteners printed on each sample photo in `images/` | 3 photos, 5 sweeteners |

A test checks that the split reproduces the notebook's exact lists.

**Why the extra sets:** every test ingredient also appears in the training lists, so the test set measures memory. The unseen and challenge sets measure recognition of new names.
