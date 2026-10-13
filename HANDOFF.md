# Sweet Truth web app: handoff

Status as of 14 October 2026. Everything below lives on the `web-app` branch; `main` is untouched.

## Where things stand

The working version of the web app is done: a visitor can type an ingredient list or photograph a pack and see the sweeteners highlighted. It runs locally only. It has not been restyled or put online yet.

| Phase | What | Status |
| --- | --- | --- |
| 1 | Detector as a reusable package (`src/sweet_truth/`) | Done |
| 2 | Web API and command line tool | Done |
| 3 | Working web page (plain styling) | Done |
| 4 | Photo quality fixes | Done |
| 5 | Portfolio-themed design | **Next** |
| 6 | Hosting, so the public can use it | After 5 |
| Later | Model improvements; live camera scan | Not started, independent of each other |

## Run it

```bash
cd /Users/bhawakshipunia/data_science/portfolio_projects/sweet_truth
uv run sweet-truth serve                  # then open http://127.0.0.1:8000
uv run sweet-truth serve --host 0.0.0.0   # to open it from a phone on the same Wi-Fi
```

For the phone, open `http://<Mac's Wi-Fi address>:8000`; `ipconfig getifaddr en0` prints the address. It must be `http://`, not `https://`.

Other commands:

```bash
uv run pytest                              # 25 tests, about 50 seconds
uv run ruff check                          # linter
uv run sweet-truth text "Glucose, Honey, Salt"
uv run sweet-truth image images/toothpaste.jpg
uv run sweet-truth evaluate images/        # one-line result for every image in a folder
```

If `uv` is not found, open a new terminal (it was installed to `~/.local/bin` and is loaded from `~/.zshrc`).

## How it is put together

- `src/sweet_truth/text.py`: splits an ingredient list, cleans each ingredient and classifies it. Keeps the position of each ingredient in the original text, which is what lets the page highlight it.
- `src/sweet_truth/ocr.py`: reads words and their positions from an image with Tesseract, in-process (`tesserocr`, no subprocess).
- `src/sweet_truth/image.py`: photo to ingredient list to sweeteners, plus the boxes to draw.
- `src/sweet_truth/api.py`: FastAPI app. `GET /api/health`, `POST /api/detect/text`, `POST /api/detect/image`, and it serves the page in `web/`.
- `src/sweet_truth/cli.py`: the `sweet-truth` command (Typer).
- `src/sweet_truth/config.py`: limits and paths, overridable with `SWEET_TRUTH_*` environment variables. A retrained model is a one-file swap via `SWEET_TRUTH_MODEL_PATH` or by replacing the pickle.
- `src/module_detector.py`: the functions the notebooks use, same names and outputs as before, now thin wrappers over the package.
- `web/`: `index.html`, `styles.css`, `app.js`. No build step. All colours and fonts are CSS variables at the top of `styles.css`, ready for the restyle.

On the page, the Photo tab works in three steps: take or choose a photo, crop it to the ingredient list on a full-screen crop view, then see the result with boxes on the photo. The text read from the photo is shown in an editable box with a Re-check button.

## What changed compared with the notebooks

The typed-text detector still flags exactly the same sweeteners as the original notebook code on all 1,000 lists in `data/ingredients_sweeteners_list.csv` (there is a test for this). The differences are:

- Photos go to Tesseract as plain grayscale, not blurred and thresholded at 128. On the 8 sample images this read better, and `sugar_free_chocolate.jpg` now works. Eight images is a small sample.
- Boxes are drawn from the exact words each sweetener was read from. This removed the "Sodium" / "Trisodium" over-annotation.
- Mismatched brackets from OCR, such as `(472e]`, are tolerated.
- "and" / "or" end an ingredient, so "Sugar and Invert Syrup" gives two items.
- A photo with no "Ingredients" heading is checked in full, with a note saying so.
- `annotate_image_with_sweeteners` saves its image with correct colours (it used to swap red and blue).

## Known gaps

- Text printed after the ingredient list (for example "CONTAINS NATURALLY OCCURRING SUGARS") is treated as an ingredient and can be flagged. Cropping avoids it; a rule for where the list ends has not been decided.
- Tested on Android-style Chrome emulation and on your own phone. Not tested on an iPhone's Safari.
- `pip install -r requirements.txt` was not run; only the `uv` path was.
- No request rate limit yet. It only matters once the app is public.
- The crop frame appears after the shot, not inside the camera view. A frame in a live viewfinder needs HTTPS, so it comes with hosting.

## Next: Phase 5, design

Agreed scope:

- Restyle `web/` after the portfolio at `/Users/bhawakshipunia/data_science/portfolio/index.html`: its dark (violet) and light (cream) themes with the same toggle, Lora for headings, Plus Jakarta Sans for body, Caveat for handwritten hints, 2 to 4 px corners, and its `--marker` highlighter colour for sweeteners.
- An about-the-author block.
- A "built with Claude" reveal in the footer.
- A share-preview image and tags so links look good when posted.
- No behaviour changes in this phase.

## Then: Phase 6, going public

- Needs a hosting account from you. Front-runner is Hugging Face Spaces (free; sleeps after about 48 hours idle, then the next visitor waits 30 to 60 seconds, to be covered by a "waking up" screen). Alternatives: Render, Fly.io.
- Add a `Dockerfile` (Python 3.12, Tesseract and its English data, the package from `pyproject.toml`).
- Add the rate limit.
- Update this repo's README with a "Try it live" link and a demo GIF.
- Update the portfolio: add a live-app link to the Sweet Truth entry in `index.html` (the `key:'swt'` entry) and to `projects/sweet-truth-detector.html`.
- Merge `web-app` into `main`. This file can be deleted at that point.

## Later ideas

- **Model:** more labelled data, a larger alias list, classifying ingredients inside brackets (currently dropped), and optionally a "this is wrong" button to collect corrections.
- **Live camera scan:** in-page camera with a frame in the viewfinder, reading the label continuously. Needs HTTPS, so after hosting.

## To pick this up with Claude

Open the project in Claude Code and say: "Read HANDOFF.md and continue with Phase 5." The full original plan is at `~/.claude/plans/i-want-to-make-misty-piglet.md`.
