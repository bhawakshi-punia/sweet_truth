"use strict";

// Photos are shrunk in the browser before upload: faster on mobile data,
// and drawing to a canvas applies the phone's rotation.
const MAX_IMAGE_SIDE = 2400;
const SVG_NS = "http://www.w3.org/2000/svg";

const $ = (selector, root = document) => root.querySelector(selector);

const PHOTO_PROBLEMS = {
  invalid_image: "That file could not be read as an image. Try another photo.",
  no_text_found:
    "No text could be read from this photo. Keep the pack flat and well lit, and move closer.",
};
const NO_HEADING_NOTE =
  'The word "Ingredients" was not found in this photo, so all the text in it was checked.';

// ---------- talking to the server ----------

// Where the detector runs. Empty means the same site as this page; the published page
// sets it in config.js, because the page and the detector are hosted separately.
const API_BASE = (window.SWEET_TRUTH_API || "").replace(/\/$/, "");
const OFFLINE_MESSAGE = "The detector is not reachable. Check your connection and try again.";

class ApiError extends Error {
  constructor(message, body = {}) {
    super(message);
    this.code = body.code;
  }
}

async function request(path, options) {
  if (!(await detectorAwake())) throw new ApiError(OFFLINE_MESSAGE);
  let response;
  try {
    response = await fetch(API_BASE + path, options);
  } catch {
    throw new ApiError(OFFLINE_MESSAGE);
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : null;
    throw new ApiError(body.message || detail || "Something went wrong. Please try again.", body);
  }
  return body;
}

const detectText = (text) =>
  request("/api/detect/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

function detectImage(blob) {
  const form = new FormData();
  form.append("image", blob, "label.jpg");
  return request("/api/detect/image", { method: "POST", body: form });
}

// ---------- showing results ----------

function showStatus(element, message, isError = false) {
  element.textContent = message;
  element.classList.toggle("error", isError);
  element.hidden = !message;
}

function summarise(sweeteners) {
  if (sweeteners.length === 0) return "No sweeteners detected";
  return sweeteners.length === 1 ? "1 sweetener found" : `${sweeteners.length} sweeteners found`;
}

/** Write `text` into `element`, marking the sweeteners among `items`. */
function renderHighlighted(element, text, items) {
  element.replaceChildren();
  let cursor = 0;
  for (const item of items.filter((i) => i.is_sweetener).sort((a, b) => a.start - b.start)) {
    if (item.start < cursor) continue;
    element.append(text.slice(cursor, item.start));
    const mark = document.createElement("mark");
    mark.textContent = text.slice(item.start, item.end);
    mark.title = `${Math.round(item.confidence * 100)}% confident`;
    element.append(mark);
    cursor = item.end;
  }
  element.append(text.slice(cursor));
}

function renderSweeteners(list, sweeteners) {
  list.replaceChildren(
    ...sweeteners.map((item) => {
      const row = document.createElement("li");
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = item.text;
      const confidence = document.createElement("span");
      confidence.className = "confidence";
      confidence.textContent = `${Math.round(item.confidence * 100)}% confident`;
      row.append(name, confidence);
      if (item.reasons && item.reasons.length) {
        // Which letter chunks made the model think "sweetener", e.g. "-ose" or "malt".
        const why = document.createElement("span");
        why.className = "why";
        why.textContent = `because of: ${item.reasons.join(", ")}`;
        row.append(why);
      }
      return row;
    }),
  );
  list.hidden = sweeteners.length === 0;
}

function renderBoxes(svg, result) {
  svg.setAttribute("viewBox", `0 0 ${result.image_width} ${result.image_height}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.replaceChildren(
    ...result.boxes.map((box) => {
      const rect = document.createElementNS(SVG_NS, "rect");
      rect.setAttribute("x", box.x1);
      rect.setAttribute("y", box.y1);
      rect.setAttribute("width", box.x2 - box.x1);
      rect.setAttribute("height", box.y2 - box.y1);
      return rect;
    }),
  );
}

/** Disable the page's controls while `work` runs. */
async function busy(statusElement, message, work) {
  const controls = document.querySelectorAll("main button:not(#theme-toggle), main input, main textarea");
  const wasDisabled = new Map([...controls].map((c) => [c, c.disabled]));
  controls.forEach((control) => (control.disabled = true));
  document.querySelectorAll("label.button").forEach((l) => l.setAttribute("aria-disabled", "true"));
  showStatus(statusElement, message);
  try {
    await work();
  } finally {
    controls.forEach((control) => (control.disabled = wasDisabled.get(control)));
    document.querySelectorAll("label.button").forEach((l) => l.removeAttribute("aria-disabled"));
  }
}

// ---------- tabs ----------

const tabs = [$("#tab-photo"), $("#tab-text")];

function selectTab(selected) {
  for (const tab of tabs) {
    const isSelected = tab === selected;
    tab.setAttribute("aria-selected", String(isSelected));
    tab.tabIndex = isSelected ? 0 : -1;
    $(`#${tab.getAttribute("aria-controls")}`).hidden = !isSelected;
  }
}

tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    const next = tabs[(index + 1) % tabs.length];
    selectTab(next);
    next.focus();
  });
});

// ---------- Type tab ----------

const textInput = $("#text-input");
const textStatus = $("#text-status");
const textResult = $("#text-result");

async function runText() {
  const text = textInput.value;
  textResult.hidden = true;
  if (!text.trim()) {
    showStatus(textStatus, "Type or paste an ingredient list first.", true);
    return;
  }
  await busy(textStatus, "Checking the ingredients…", async () => {
    try {
      const result = await detectText(text);
      $(".summary", textResult).textContent = summarise(result.sweeteners);
      renderHighlighted($(".highlighted", textResult), result.text, result.items);
      renderSweeteners($(".sweeteners", textResult), result.sweeteners);
      textResult.hidden = false;
      showStatus(textStatus, "");
    } catch (error) {
      showStatus(textStatus, error.message, true);
    }
  });
}

$("#text-form").addEventListener("submit", (event) => {
  event.preventDefault();
  runText();
});

$("#text-example").addEventListener("click", () => {
  textInput.value = textInput.placeholder;
  runText();
});

// ---------- Photo tab ----------

const photoStatus = $("#photo-status");
const photoResult = $("#photo-result");
const photoText = $("#photo-text");
const photoImage = $(".annotated img", photoResult);
const recheckHighlighted = $(".highlighted", $("#recheck-form"));

/** Load an image file onto a canvas no larger than MAX_IMAGE_SIDE. */
async function prepareImage(blob) {
  const url = URL.createObjectURL(blob);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(image.naturalWidth, image.naturalHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(image.naturalWidth * scale);
    canvas.height = Math.round(image.naturalHeight * scale);
    canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas;
  } catch {
    throw new ApiError(PHOTO_PROBLEMS.invalid_image);
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** Turn a canvas, or the `area` of it given in fractions of its size, into a JPEG. */
function toJpeg(canvas, area = { x: 0, y: 0, width: 1, height: 1 }) {
  const out = document.createElement("canvas");
  out.width = Math.max(1, Math.round(canvas.width * area.width));
  out.height = Math.max(1, Math.round(canvas.height * area.height));
  out
    .getContext("2d")
    .drawImage(canvas, -Math.round(canvas.width * area.x), -Math.round(canvas.height * area.y));
  return new Promise((resolve, reject) =>
    out.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new ApiError(PHOTO_PROBLEMS.invalid_image))),
      "image/jpeg",
      0.92,
    ),
  );
}

function showPhotoResult(sweeteners) {
  $(".summary", photoResult).textContent = summarise(sweeteners);
  renderSweeteners($(".sweeteners", photoResult), sweeteners);
  photoResult.hidden = false;
}

/** Send the photo for checking and show what comes back. `getJpeg` produces the upload. */
async function runPhoto(getJpeg) {
  photoResult.hidden = true;
  await busy(photoStatus, "Reading the label…", async () => {
    try {
      const upload = await getJpeg();
      const result = await detectImage(upload);

      if (photoImage.src) URL.revokeObjectURL(photoImage.src);
      photoImage.src = URL.createObjectURL(upload);
      renderBoxes($(".annotated svg", photoResult), result);
      photoText.value = result.ingredients_text;
      recheckHighlighted.hidden = true;
      showPhotoResult(result.sweeteners);
      $("#photo-example").hidden = true;
      showStatus(photoStatus, result.heading_found ? "" : NO_HEADING_NOTE);
    } catch (error) {
      showStatus(photoStatus, PHOTO_PROBLEMS[error.code] || error.message, true);
    }
  });
}

// ---------- framing the ingredient list on a chosen photo ----------

const photoCrop = $("#photo-crop");
const cropImage = $(".cropper img", photoCrop);
const cropFrame = $(".crop-frame", photoCrop);
const MIN_CROP = 0.08; // smallest frame, as a fraction of the photo

let cropCanvas = null;
// The frame, in fractions of the photo: left, top, right, bottom.
let crop = { left: 0, top: 0, right: 1, bottom: 1 };

function drawCropFrame() {
  cropFrame.style.left = `${crop.left * 100}%`;
  cropFrame.style.top = `${crop.top * 100}%`;
  cropFrame.style.width = `${(crop.right - crop.left) * 100}%`;
  cropFrame.style.height = `${(crop.bottom - crop.top) * 100}%`;
}

const cropAgain = $("#crop-again");

/** Open the full-screen crop step. With a `file`, start on that new photo. */
async function openCropper(file) {
  if (file) {
    photoResult.hidden = true;
    showStatus(photoStatus, "");
    try {
      cropCanvas = await prepareImage(file);
      if (cropImage.src) URL.revokeObjectURL(cropImage.src);
      cropImage.src = URL.createObjectURL(await toJpeg(cropCanvas));
      await cropImage.decode();
    } catch {
      cropCanvas = null;
      cropAgain.hidden = true;
      showStatus(photoStatus, PHOTO_PROBLEMS.invalid_image, true);
      return;
    }
    crop = { left: 0.05, top: 0.05, right: 0.95, bottom: 0.95 };
    $("#photo-example").hidden = true;
  }
  drawCropFrame();
  cropAgain.hidden = true;
  photoCrop.showModal();
}

// Closing the crop step (button, Esc or the phone's back gesture) leaves a way back in.
photoCrop.addEventListener("close", () => (cropAgain.hidden = !cropCanvas));
$("#crop-cancel").addEventListener("click", () => photoCrop.close());
cropAgain.addEventListener("click", () => openCropper());

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

// Dragging the frame moves it; dragging a corner resizes it.
cropFrame.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  const corner = event.target.dataset.corner; // undefined when the frame itself is grabbed
  const bounds = cropImage.getBoundingClientRect();
  const start = { ...crop, x: event.clientX, y: event.clientY };
  cropFrame.setPointerCapture(event.pointerId);

  const onMove = (move) => {
    const dx = (move.clientX - start.x) / bounds.width;
    const dy = (move.clientY - start.y) / bounds.height;
    if (!corner) {
      const width = start.right - start.left;
      const height = start.bottom - start.top;
      crop.left = clamp(start.left + dx, 0, 1 - width);
      crop.top = clamp(start.top + dy, 0, 1 - height);
      crop.right = crop.left + width;
      crop.bottom = crop.top + height;
    } else {
      if (corner.includes("w")) crop.left = clamp(start.left + dx, 0, crop.right - MIN_CROP);
      if (corner.includes("e")) crop.right = clamp(start.right + dx, crop.left + MIN_CROP, 1);
      if (corner.includes("n")) crop.top = clamp(start.top + dy, 0, crop.bottom - MIN_CROP);
      if (corner.includes("s")) crop.bottom = clamp(start.bottom + dy, crop.top + MIN_CROP, 1);
    }
    drawCropFrame();
  };
  const onEnd = () => {
    cropFrame.removeEventListener("pointermove", onMove);
    cropFrame.removeEventListener("pointerup", onEnd);
    cropFrame.removeEventListener("pointercancel", onEnd);
  };
  cropFrame.addEventListener("pointermove", onMove);
  cropFrame.addEventListener("pointerup", onEnd);
  cropFrame.addEventListener("pointercancel", onEnd);
});

$("#crop-confirm").addEventListener("click", () => {
  const area = {
    x: crop.left,
    y: crop.top,
    width: crop.right - crop.left,
    height: crop.bottom - crop.top,
  };
  photoCrop.close();
  runPhoto(() => toJpeg(cropCanvas, area));
});

$("#crop-skip").addEventListener("click", () => {
  photoCrop.close();
  runPhoto(() => toJpeg(cropCanvas));
});

for (const input of [$("#photo-camera"), $("#photo-gallery")]) {
  input.addEventListener("change", () => {
    const [file] = input.files;
    input.value = ""; // so picking the same photo again still fires "change"
    if (file) openCropper(file);
  });
}

// The example photo is already framed, so it is checked straight away.
$("#photo-example").addEventListener("click", () => {
  const source = $("#photo-example img").src;
  cropCanvas = null;
  cropAgain.hidden = true;
  runPhoto(async () => {
    const response = await fetch(source).catch(() => null);
    if (!response || !response.ok) throw new ApiError("The example photo could not be loaded.");
    return toJpeg(await prepareImage(await response.blob()));
  });
});

// Let people correct what was read from the photo and check that text instead.
$("#recheck-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = photoText.value;
  if (!text.trim()) return;
  await busy(photoStatus, "Checking the ingredients…", async () => {
    try {
      const result = await detectText(text);
      showPhotoResult(result.sweeteners);
      renderHighlighted(recheckHighlighted, result.text, result.items);
      recheckHighlighted.hidden = false;
      showStatus(photoStatus, "");
    } catch (error) {
      showStatus(photoStatus, error.message, true);
    }
  });
});

// ---------- theme ----------

const themeToggle = $("#theme-toggle");

function paintThemeToggle() {
  const dark = document.documentElement.classList.contains("dark");
  $(".theme-icon", themeToggle).textContent = dark ? "☀️" : "🌙";
  $(".theme-label", themeToggle).textContent = dark ? "Light" : "Dark";
  themeToggle.setAttribute("aria-label", `Switch to ${dark ? "light" : "dark"} theme`);
}

themeToggle.addEventListener("click", () => {
  const dark = document.documentElement.classList.toggle("dark");
  try {
    localStorage.setItem("theme", dark ? "dark" : "light");
  } catch {
    // Storage can be unavailable (private browsing); the theme then lasts until reload.
  }
  paintThemeToggle();
});
paintThemeToggle();

// ---------- waking the detector ----------

// The free hosting puts the detector to sleep when nobody has used it for a while, and the
// first visit after that wakes it, which takes up to a minute. Say so instead of looking broken.
const WAKING_HINT_DELAY = 1500; // ms before a slow answer counts as "waking up"
const WAKING_POLL_INTERVAL = 4000;
const WAKING_GIVE_UP = 4 * 60 * 1000;

const waking = $("#waking");
let wakingClock = null;

function showWaking(show) {
  if (show === !waking.hidden) return;
  waking.hidden = !show;
  clearInterval(wakingClock);
  if (!show) return;
  const started = Date.now();
  const elapsed = $(".waking-elapsed", waking);
  elapsed.textContent = "";
  wakingClock = setInterval(() => {
    elapsed.textContent = `${Math.round((Date.now() - started) / 1000)} s`;
  }, 1000);
}

async function healthy() {
  try {
    const response = await fetch(`${API_BASE}/api/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(15000),
    });
    return response.ok && (await response.json()).status === "ok";
  } catch {
    return false; // still asleep or starting: no answer, or the host's own holding page
  }
}

async function waitForDetector() {
  const started = Date.now();
  const hint = setTimeout(() => showWaking(true), WAKING_HINT_DELAY);
  try {
    while (Date.now() - started < WAKING_GIVE_UP) {
      if (await healthy()) {
        $("#offline").hidden = true;
        return true;
      }
      showWaking(true);
      await new Promise((resolve) => setTimeout(resolve, WAKING_POLL_INTERVAL));
    }
    $("#offline").hidden = false;
    return false;
  } finally {
    clearTimeout(hint);
    showWaking(false);
  }
}

let detectorCheck = waitForDetector();

/** Resolves true once the detector answers; checks again if it had given up before. */
async function detectorAwake() {
  if (!(await detectorCheck)) detectorCheck = waitForDetector();
  return detectorCheck;
}
