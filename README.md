# The Sweet Truth Detector

A machine learning pipeline that detects hidden sweeteners in packaged food ingredient lists. It combines OCR, text preprocessing and a classifier to find and highlight sweeteners, from typed text or from a photo of the packaging.

**Try it live:** [bhawakshi-punia.github.io/sweet_truth](https://bhawakshi-punia.github.io/sweet_truth/), on your phone or computer

**Case study:** [bhawakshi-punia.github.io/#projects/sweet-truth](https://bhawakshi-punia.github.io/#projects/sweet-truth)

![Annotated ingredient list with detected sweeteners](https://github.com/user-attachments/assets/281466a3-422c-4427-bff0-44b2e65361a7)

## Why this project

From invert syrup to maltitol, sweeteners often appear on labels under unfamiliar names. This tool reads an ingredient list and points out every ingredient that is a sweetener, natural or artificial.

## How it works

The detector has two parts:

1. **Sweetener detection from text**: the list is split into single ingredients (keeping bracketed codes such as "(955)"). Each ingredient is turned into TF-IDF features over letter chunks of 1 to 6 characters and classified as sweetener or not by logistic regression. Letter chunks let it recognise names it has never seen, from parts such as *-ose*, *-itol* or *syrup*, and show why it flagged an ingredient. This is **model v2**; the original word-level random forest (**v1**) is kept for the notebooks.
2. **OCR integration**: an image of the ingredient list is loaded with OpenCV, the text is extracted with Tesseract OCR and passed to the classifier, and detected sweeteners are marked on the image with bounding boxes.

## Data

- [`data/final_ingredient_dataset.csv`](data/final_ingredient_dataset.csv): 392 individual ingredients labelled as sweetener (1) or non-sweetener (0). Curated for commonly available Indian food products from the [FDA list of food ingredient types](https://www.fda.gov/food/food-additives-and-gras-ingredients-information-consumers/types-food-ingredients), a [Mendeley dataset](https://data.mendeley.com/datasets/58mfpfxksk/1/files/3f0a044f-0703-46a4-a59b-08e7ace09b3e) and articles listing common sweeteners.
- [`data/ingredients_sweeteners_list.csv`](data/ingredients_sweeteners_list.csv): 1,000 synthetic ingredient lists built from the dataset above, used to train and evaluate the classifier.

## Results

**Model v2 (in the app).** On ingredients never seen in training, v2 finds **87%** of sweeteners, against 38% for v1, with 0.92 precision. It finds all 10 sweeteners printed on the sample photos with no false flags. It is 0.2 MB in size.

![Confusion matrices for v1 and v2 on every evaluation set](reports/model_v2/figures/confusion_overview.png)

Full comparison, including the transformer experiment (ELECTRA-small) and which change mattered most: [`reports/model_v2/`](reports/model_v2/README.md). How performance is judged: [EVALUATION.md](reports/model_v2/EVALUATION.md). Training data and approach: [`src/sweet_truth/training/`](src/sweet_truth/training/README.md).

**Model v1 (original notebook).** The tuned random forest (`n_estimators=451`, `min_samples_split=6`) reaches **97% accuracy** on the hold-out set of 100 ingredient lists, deciding whether each list contains a sweetener. All hold-out ingredients also appear in the training lists, so this mostly measures remembering names. On unseen names v1 finds only 38% of sweeteners.

| | |
| --- | --- |
| ![Hold-out confusion matrix](images/hold_out_confusion_matrix.png) | ![Toothpaste label with sweeteners annotated](images/toothpaste_annotated.jpg) |
| v1: hold-out set confusion matrix (lists) | A toothpaste label with detected sweeteners boxed |

## Getting started

```bash
git clone https://github.com/bhawakshi-punia/sweet_truth.git
cd sweet_truth
uv sync --all-extras
```

This uses [uv](https://docs.astral.sh/uv/) and installs everything, including the spaCy model, into `.venv`. Without uv, `pip install -r requirements.txt` in a virtual environment does the same.

Reading text from photos needs Tesseract's English language data (`eng.traineddata`). Installing [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) provides it (for example `brew install tesseract` on macOS or `apt install tesseract-ocr` on Debian/Ubuntu). If it is somewhere unusual, set `SWEET_TRUTH_TESSDATA_DIR` to the folder that contains it.

## Web app

The app is live at [bhawakshi-punia.github.io/sweet_truth](https://bhawakshi-punia.github.io/sweet_truth/). The page is served by GitHub Pages and the detector runs on Render's free tier, which sleeps when unused, so the first check after a quiet spell can take up to a minute while it wakes. Both redeploy automatically on every push to `main`.

To run it on your own computer:

```bash
uv run sweet-truth serve
```

Then open http://127.0.0.1:8000. Type or paste an ingredient list, or take or upload a photo of a pack: the sweeteners are highlighted in the text and boxed on the photo. Photos are processed in memory and are not stored.

To try it from a phone on the same Wi-Fi, run `uv run sweet-truth serve --host 0.0.0.0` and open `http://<your computer's IP address>:8000` on the phone.

The same detector is available from the command line:

```bash
uv run sweet-truth text "Glucose, Honey, Salt, Vitamins, Minerals"
uv run sweet-truth image images/children_nutrition_drink.jpg
uv run sweet-truth evaluate images/     # summary for every image in a folder
```

## Usage in Python

Run the following from the repository root. These functions use model v1, as the notebooks do; the app and the `sweet-truth` command use model v2.

### Step 1: Import the module and load the model

```python
import sys
sys.path.append("src")

from module_detector import *
import joblib

model = joblib.load("models/sweet_truth_detector.pkl")
```

### Step 2: Detect sweeteners from text

`sweet_truth_from_list(ingredient_list, model)` returns a dataframe of the detected sweeteners and the model's confidence in each.

```python
ingredient_list = "Glucose, Honey, Salt, Vitamins, Minerals"
sweet_truth_from_list(ingredient_list, model)
```

### Step 3: Detect sweeteners from an image

Give the path to a clear image of an ingredient list. Example images are in [`images/`](images/).

```python
image_path = "images/children_nutrition_drink.jpg"

# results as a dataframe
sweet_truth_from_image(image_path, model)

# annotate the image with bounding boxes around detected sweeteners
annotate_image_with_sweeteners(image_path, model)
```

The annotated image is saved next to the input, so `image.jpg` produces `image_annotated.jpg`.

## Repository layout

```
├── data/
│   ├── final_ingredient_dataset.csv        # individual ingredients, labelled
│   ├── challenge_ingredients.csv           # real names not in the data, for testing
│   ├── photo_sweeteners.csv                # sweeteners printed on each sample photo
│   └── ingredients_sweeteners_list.csv     # synthetic ingredient lists for training
├── notebooks/
│   ├── sweetener_detector_from_text.ipynb  # Part 1: build and test the text classifier
│   ├── ocr.ipynb                           # Part 2: OCR on ingredient images
│   ├── the_sweet_truth_detector.ipynb      # Parts 1 + 2: full pipeline, image to annotated output
│   ├── model_v2_eda.ipynb                  # model v2: data exploration
│   ├── model_v2_evaluation.ipynb           # model v2: results, mistakes, try it yourself
│   └── model_v2_colab_train.ipynb          # model v2: training on a Colab GPU
├── src/
│   ├── sweet_truth/                        # the detector as a package
│   │   ├── text.py                         # split, clean and classify an ingredient list
│   │   ├── ocr.py                          # read words from an image with Tesseract
│   │   ├── image.py                        # photo to sweeteners and their positions
│   │   ├── api.py                          # FastAPI app serving the API and the web page
│   │   ├── cli.py                          # the sweet-truth command
│   │   ├── classifiers.py                  # one interface for all ingredient classifiers
│   │   ├── training/                       # model v2: data, training, evaluation
│   │   └── config.py, schemas.py, errors.py
│   └── module_detector.py                  # notebook-friendly functions built on the package
├── web/                                    # the web page (plain HTML, CSS and JavaScript)
├── tests/                                  # pytest suite
├── reports/model_v2/                       # v1 vs v2 comparison: results, figures, how it is judged
├── models/
│   ├── sweet_truth_v2.joblib               # model v2 (used by the app)
│   └── sweet_truth_detector.pkl            # model v1 (used by the notebooks)
├── images/                                 # example inputs and annotated outputs
├── pyproject.toml, uv.lock                 # dependencies
├── Dockerfile, render.yaml                # how the detector is hosted on Render
├── .github/workflows/pages.yml           # publishes web/ to GitHub Pages
└── requirements.txt
```

Run the tests with `uv run pytest` and the linter with `uv run ruff check`.

The notebooks use paths relative to `notebooks/`, so run them from that folder. The model was saved with scikit-learn 1.5.2; loading it with another version may raise warnings.

## Limitations and next steps

- Text printed after the ingredient list (for example "Contains naturally occurring sugars") is read as part of the list and can be flagged.
- Results depend on a clear, well-lit photo of the ingredient list.
- Sweeteners whose names give no clue (thaumatin, cyclamate) are missed; look-alikes such as ethyl maltol are flagged.
- Next: a curated list of sweeteners and their INS / E-numbers, more labelled ingredient names, and live camera scanning.

## Tech stack

Python · scikit-learn (TF-IDF, logistic regression, random forest) · spaCy · regular expressions · Tesseract OCR · OpenCV · FastAPI · Typer · Jupyter

## Image sources

The images used for testing in `the_sweet_truth_detector.ipynb` and `ocr.ipynb` are:

- **Self-captured**: images that I photographed for this project.
- **Open Food Facts**: openly available food ingredient images from [Open Food Facts](https://world.openfoodfacts.org/). These images are used solely for demonstration purposes and align with Open Food Facts' terms of use.
