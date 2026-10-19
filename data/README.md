# Data

Licensed [CC BY 4.0](LICENSE); please credit the project if you reuse it.

| File | What it is |
| --- | --- |
| `final_ingredient_dataset.csv` | 392 single ingredients labelled sweetener (1) or not (0), curated for Indian packaged foods from the FDA list of food ingredient types, the Food Products Ingredient Dataset (Kumar & Singh, 2024) and articles on sweeteners |
| `ingredients_sweeteners_list.csv` | 1,000 synthetic ingredient lists built from the file above, with the sweeteners each contains (made in `notebooks/01_v1_text_classifier.ipynb`) |
| `challenge_ingredients.csv` | 46 real ingredient names not in the data, labelled by hand, for testing on unseen names |
| `photo_sweeteners.csv` | The sweeteners printed on each sample photo |

How they are used: [DATASET.md](../src/sweet_truth/training/DATASET.md).
