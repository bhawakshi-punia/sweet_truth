# Dataset

Model v1 and v2 learn from the same data. Nothing in the original CSVs was changed.

## The two source files

| File | Rows | What it holds |
| --- | --- | --- |
| [`data/final_ingredient_dataset.csv`](../../../data/final_ingredient_dataset.csv) | 392 | Single ingredients labelled `1` (sweetener) or `0` (anything else). Curated for Indian packaged foods from the FDA list of food ingredient types, a Mendeley dataset and articles on sweeteners. |
| [`data/ingredients_sweeteners_list.csv`](../../../data/ingredients_sweeteners_list.csv) | 1,000 | Synthetic ingredient lists: 3 to 7 rows of the file above, picked at random and joined with ", ". Each row also names the sweeteners in it (`nil` if none). |

### Class balance

- **Ingredients:** 53 of the 392 are sweeteners (14%). Across the occurrences in the lists, 13% are sweeteners.
- **Lists:** 491 of the 1,000 contain at least one sweetener.

### Variants and one label conflict

- **Variants:** some ingredients appear in several spellings, such as "Maltitol", "MALTITOL (965)" and "maltitol". Ignoring case and bracketed notes, the 392 rows are 360 distinct ingredients.
- **Conflict:** "Maltodextrin" is labelled a sweetener but "MALTODEXTRIN" is not. This causes the few false positives every model makes on the training set. The data was left as it is, so the comparison uses exactly the original data.
- **Multi-ingredient rows:** some rows hold more than one ingredient, such as "fruit and vegetable juices" and "RAISING NTS (INS 503(ii), ...), COCOA SOLIDS (0.7%)".

## From lists to labelled ingredients

Every list splits back exactly into the rows it was built from ([`data.py`](data.py), `decompose`). Some row names contain ", " themselves, so pieces are joined until they form a known name. This gives every ingredient in every list an exact label. Model v2 is trained on those single ingredients.

## Splits

| Set | What | Size |
| --- | --- | --- |
| **Train** | The notebook's training lists, split into ingredients | 800 lists, 4,035 ingredient occurrences |
| **Validation** | The notebook's validation lists | 100 lists, 512 ingredients |
| **Test** | The notebook's hold-out lists | 100 lists, 506 ingredients |
| **Unseen ingredients** | 5-fold cross-validation over the 392 rows, grouped so that all spellings of an ingredient fall in the same fold, and stratified so each fold has sweeteners | about 78 names per fold, 391 in total (the CSV lists "CHOCO CREAM" twice) |
| **Challenge** | [`data/challenge_ingredients.csv`](../../../data/challenge_ingredients.csv): 46 real ingredient names that are not in the data, labelled by hand (24 sweeteners, 22 not, including look-alikes such as sorbic acid and ethyl maltol) | 46 |
| **Photos** | [`data/photo_sweeteners.csv`](../../../data/photo_sweeteners.csv): the sweeteners printed on each sample photo in `images/` | 8 photos, 10 sweeteners |

- **Train, validation and test** are the exact split of the original notebook: `train_test_split(..., test_size=0.2, random_state=101)`, then the 20% halved with the same seed. A test checks that the same lists come out.
- **Why the extra sets:** all 506 test ingredients also appear in the training lists, because every list was drawn from the same 392 rows. The test set therefore mostly measures whether a model remembers names. The unseen-ingredient folds and the challenge set measure whether it can recognise names it has never seen.
