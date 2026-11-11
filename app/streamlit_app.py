"""AgriCL-Net web app: leaf photo -> disease, severity, affected area and advice.

Run:  streamlit run app/streamlit_app.py
Models come from a bundle folder: a full bundle (notebook 05) or an edge
bundle (notebook 06). Without one the app runs in DEMO mode, which only
previews the interface and is not a diagnosis.
Environment: AGRICL_BUNDLE (bundle folder), AGRICL_DEMO=1 (force demo).
Icons are Google Material Symbols (built into Streamlit); no emojis.
Layout works from phone width up: columns stack below 640 px.
"""

from __future__ import annotations

import hashlib
import io
import os
import random
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agricl.demo import DemoPredictor  # noqa: E402

DEFAULT_BUNDLE = ROOT / "models" / "bundle"
PLANTSEG = ROOT / "data" / "plantseg"
UNKNOWN_DIR = ROOT / "unknown_images"

# decision status -> (alert kind, icon, title)
STATUS = {
    "disease": ("error", ":material/report:", "Disease identified"),
    "unknown": ("warning", ":material/help:", "Unknown disease"),
    "uncertain": ("info", ":material/question_mark:", "Uncertain result"),
    "no_disease": ("success", ":material/check_circle:", "No diseased area found"),
}
LEVEL_COLOUR = {"Mild": "green", "Moderate": "orange", "Severe": "red", "No leaf": "gray"}
SOURCES = {"Upload": ":material/upload: Upload", "Camera": ":material/photo_camera: Camera",
           "Example": ":material/photo_library: Example"}

st.set_page_config(page_title="AgriCL-Net", page_icon=":material/eco:", layout="wide",
                   initial_sidebar_state="auto")

# Phone-friendly spacing and type; hide the animated "running" figure (the
# spinner below shows progress instead).
st.markdown("""
<style>
[data-testid="stStatusWidget"] { visibility: hidden; }
img { max-width: 100%; height: auto; }
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"], .block-container { padding: 1rem 0.75rem 3rem 0.75rem; }
  h1 { font-size: 1.6rem !important; }
  h3 { font-size: 1.15rem !important; }
  [data-testid="stMetricValue"] { font-size: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models (first time only)...")
def load_predictor(bundle: str, leaf_method: str):
    from agricl.predictor import AgriCLPredictor

    return AgriCLPredictor(bundle, leaf_method=leaf_method)


@st.cache_resource(show_spinner="Loading edge models (first time only)...")
def load_edge_predictor(bundle: str):
    from agricl.edge import EdgePredictor

    return EdgePredictor(bundle)


def get_predictor(bundle: str, leaf_method: str):
    """(predictor, problem). A full bundle (notebook 05) or an edge bundle
    (notebook 06); otherwise the demo predictor, with the reason."""
    if os.environ.get("AGRICL_DEMO") == "1":
        return DemoPredictor(), "Demo mode was requested (AGRICL_DEMO=1)."
    b = Path(bundle)
    try:
        if (b / "agricl_bundle.json").exists():
            return load_predictor(bundle, leaf_method), None
        if (b / "edge_bundle.json").exists():
            return load_edge_predictor(bundle), None
    except Exception as e:  # missing packages, wrong files: show why and keep the app usable
        return DemoPredictor(), f"The model bundle could not be loaded: {e}"
    return DemoPredictor(), f"No model bundle found in {bundle}."


@st.cache_data
def example_paths(n: int = 12) -> list[str]:
    """A few PlantSeg test photos, if the dataset is on this machine."""
    test = PLANTSEG / "images" / "test"
    if not test.exists():
        return []
    files = sorted(test.glob("*.jpg"))
    return [str(p) for p in random.Random(0).sample(files, min(n, len(files)))]


def save_unknown(data: bytes) -> Path:
    """Store an image the system could not identify, for the next learning round.
    Named by content hash, so reruns of the same photo do not create copies."""
    UNKNOWN_DIR.mkdir(exist_ok=True)
    path = UNKNOWN_DIR / f"{hashlib.sha256(data).hexdigest()[:16]}.jpg"
    if not path.exists():
        path.write_bytes(data)
    return path


def alert(kind: str, text: str, icon: str) -> None:
    getattr(st, kind)(text, icon=icon)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("AgriCL-Net", icon=":material/eco:")
    st.caption("Plant disease diagnosis with severity estimation and continual learning.")
    bundle = st.text_input("Model bundle folder", os.environ.get("AGRICL_BUNDLE", str(DEFAULT_BUNDLE)))
    leaf_method = st.radio(
        "Leaf segmentation", ["sam2", "exg"],
        format_func=lambda m: {"sam2": "SAM2 (accurate)", "exg": "Colour index (fast, offline)"}[m],
    )
    explain = st.toggle("Show what the classifier looked at", value=True)

predictor, problem = get_predictor(bundle, leaf_method)
if type(predictor).__name__ == "EdgePredictor":
    st.sidebar.info("Edge models (offline): colour-index leaf mask, no heatmap.", icon=":material/wifi_off:")

# ------------------------------------------------------------------- input
st.title(":material/eco: Plant disease diagnosis")
st.write("Upload or take a photo of a leaf. AgriCL-Net finds the diseased area, estimates how severe it is, "
         "names the disease and suggests what to do.")
if problem:
    st.warning(f"**Demo mode: not a real diagnosis.** {problem} The disease names are placeholders and the "
               "diseased area is guessed from colour. Create a model bundle with notebook 05 or 06 to use the "
               "trained models.", icon=":material/science:")

source = st.segmented_control("Photo source", list(SOURCES), default="Upload", format_func=SOURCES.get)
data = None
if source == "Upload":
    f = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png", "webp"])
    data = f.getvalue() if f else None
elif source == "Camera":
    f = st.camera_input("Take a photo of the leaf")
    data = f.getvalue() if f else None
else:
    paths = example_paths()
    if not paths:
        st.info("Examples need the PlantSeg dataset in data/plantseg.", icon=":material/info:")
    else:
        choice = st.selectbox("Example photo (PlantSeg test set)", paths, format_func=lambda p: Path(p).stem)
        data = Path(choice).read_bytes()

if not data:
    st.info("Add a photo to start.", icon=":material/photo_camera:")
    st.stop()

img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
with st.spinner("Analysing the photo..."):
    d = predictor.diagnose(img, explain=explain)

# ----------------------------------------------------------------- results
for issue in d.quality.issues:
    st.warning(issue, icon=":material/photo_camera:")

kind, icon, title = STATUS[d.decision.status]
same = d.decision.message.rstrip(".") == title
alert(kind, f"**{title}.**" if same else f"**{title}.** {d.decision.message}", icon)
if d.decision.status == "unknown" and not d.demo:
    st.caption(f"Saved for the next learning round: {save_unknown(data).name}")

left, right = st.columns(2)
left.image(d.image, caption="Your photo", width="stretch")
right.image(d.overlay, caption="Green: leaf. Red: diseased area. Yellow boxes: affected regions.", width="stretch")

st.subheader(f"Disease: {d.decision.disease or 'not identified'}", icon=":material/biotech:")
c1, c2 = st.columns(2)
c1.metric("Severity", f"{d.severity.percent:.1f}%", help="Diseased area as a share of the leaf area",
          icon=":material/monitoring:", border=True)
c2.metric("Level", d.severity.level, help="Mild below 20%, Moderate 20-50%, Severe above 50%",
          icon=":material/health_and_safety:", border=True)
st.progress(min(d.severity.percent / 100, 1.0),
            text=f":{LEVEL_COLOUR.get(d.severity.level, 'gray')}[{d.severity.level}] - "
                 f"{d.severity.percent:.1f}% of the leaf is diseased, in {len(d.boxes)} affected regions")

info, cam = st.columns(2)
with info:
    st.subheader("Most likely diseases", icon=":material/query_stats:")
    for name, p in d.decision.top3:
        st.progress(p, text=f"{name}: {p * 100:.0f}%")
with cam:
    if d.cam_overlay is not None:
        st.subheader("What the classifier looked at", icon=":material/visibility:")
        st.image(d.cam_overlay, caption="Red: regions that drove the prediction", width="stretch")

st.subheader("What to do", icon=":material/tips_and_updates:")
st.write(f"Type: {d.advice.type_name}.")
st.markdown("\n".join(f"- {s}" for s in d.advice.steps))
if d.advice.severity_note:
    st.info(d.advice.severity_note, icon=":material/medical_services:")
st.caption(d.advice.disclaimer)

with st.expander("Details", icon=":material/list_alt:"):
    st.write({
        "severity_percent": round(d.severity.percent, 2),
        "leaf_pixels": d.severity.leaf_pixels,
        "lesion_pixels": d.severity.lesion_pixels,
        "classifier_crop": d.crop_box,
        "confidence": round(d.decision.confidence, 3),
        "photo_sharpness": round(d.quality.sharpness, 1),
        "photo_brightness": round(d.quality.brightness, 1),
        "seconds": {k: round(v, 2) for k, v in d.timings.items()},
    })
