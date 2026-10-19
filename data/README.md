# Data

| File | What it is | Source |
| --- | --- | --- |
| `final_ingredient_dataset.csv` | 392 single ingredients labelled sweetener (1) or not (0) | Curated for Indian packaged foods from the FDA list of food ingredient types, a Mendeley dataset and articles on sweeteners |
| `ingredients_sweeteners_list.csv` | 1,000 synthetic ingredient lists (3 to 7 rows of the file above), with the sweeteners each contains | Generated in `notebooks/sweetener_detector_from_text.ipynb` |
| `challenge_ingredients.csv` | 46 real ingredient names not in the data, labelled by hand, for testing on unseen names | Written for the model v2 comparison |
| `photo_sweeteners.csv` | The sweeteners printed on each sample photo in `images/` | Read off the photos by hand |

Details on how these are used and split: [`src/sweet_truth/training/DATASET.md`](../src/sweet_truth/training/DATASET.md).
