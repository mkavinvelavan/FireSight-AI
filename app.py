import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# FireSight AI: ViT-B/16 fire and smoke multi-label image classification.
# Model inference stays compatible with the original local / Hugging Face setup.

st.set_page_config(
    page_title="FireSight AI | Fire & Smoke Intelligence",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_ID = os.getenv("HF_MODEL_ID", "").strip()
if not MODEL_ID:
    try:
        MODEL_ID = str(st.secrets.get("HF_MODEL_ID", "")).strip()
    except Exception:
        # Local runs and Cloud deployments without secrets still use local settings.
        MODEL_ID = ""


def project_path(value):
    """Resolve configured relative paths from the project, not the shell cwd."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


LOCAL_MODEL_DIR = project_path(os.getenv("LOCAL_MODEL_DIR", "best_model"))
DEFAULT_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
ASSET_DIR = project_path("assets")
DATASET_DIR = project_path(os.getenv("DATASET_DIR", "dataset"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Results transcribed from the saved Colab test output. The confusion matrix
# image and training curves are extracted directly from the notebook outputs.
VERIFIED_RESULTS = {
    "threshold": 0.5,
    "subset_accuracy": 0.9667,
    "macro_f1": 0.9839,
    "class_report": {
        "Fire": {"precision": 0.99, "recall": 1.00, "f1": 1.00, "support": 392},
        "Smoke": {"precision": 0.98, "recall": 0.97, "f1": 0.97, "support": 424},
    },
    "confusion_matrices": {
        "Fire": {"TN": 355, "FP": 3, "FN": 0, "TP": 392},
        "Smoke": {"TN": 316, "FP": 10, "FN": 14, "TP": 410},
    },
    "dataset": {"train": 3500, "validation": 750, "test": 750, "total": 5000},
}

# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
      background: radial-gradient(ellipse at 90% 0%, rgba(127,29,29,.18), transparent 34%), #0a1020;
      color: #e6edf7;
    }
    [data-testid="stHeader"] {background: rgba(10,16,32,.9);}
    [data-testid="stSidebar"] {background:#0e1728;border-right:1px solid #26354c;}
    [data-testid="stSidebar"] * {color:#e6edf7;}
    .block-container {max-width:1400px;padding-top:1.3rem;padding-bottom:2.5rem;}
    .fs-hero {
      padding:2rem 2.2rem;border-radius:20px;
      background:linear-gradient(120deg,#172554 0%,#451a24 62%,#7c2d12 100%);
      border:1px solid rgba(255,255,255,.10);margin-bottom:1.1rem;
      box-shadow:0 18px 45px rgba(0,0,0,.22);
    }
    .fs-eyebrow {color:#fecaca;font-size:.75rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;}
    .fs-hero h1 {color:#fff;font-size:clamp(2rem,4vw,3rem);line-height:1.12;margin:.5rem 0;}
    .fs-hero p {color:#f1f5f9;font-size:1.05rem;max-width:900px;margin-bottom:.2rem;}
    .fs-muted {color:#9aabc2;font-size:.88rem;}
    .fs-card {background:linear-gradient(145deg,#142034,#0f192a);border:1px solid #26354c;border-radius:15px;padding:1rem 1.1rem;height:100%;}
    .fs-kpi-label {color:#9aabc2;font-size:.78rem;letter-spacing:.06em;text-transform:uppercase;}
    .fs-kpi-value {color:#f8fafc;font-size:1.55rem;font-weight:780;margin:.25rem 0;}
    .fs-kpi-note {color:#9aabc2;font-size:.78rem;}
    .fs-pill {display:inline-block;padding:.28rem .65rem;border-radius:99px;border:1px solid #384860;background:#17243a;color:#e2e8f0;font-size:.78rem;margin:.15rem .2rem .15rem 0;}
    .fs-callout {border-left:4px solid #ff625a;padding:.85rem 1rem;background:rgba(127,29,29,.18);border-radius:0 10px 10px 0;}
    div[data-testid="stMetric"] {background:#111b2e;border:1px solid #26354c;padding:.8rem 1rem;border-radius:13px;}
    div[data-testid="stMetricLabel"] {color:#9aabc2;}
    div[data-testid="stMetricValue"] {color:#f8fafc;}
    .stButton button[kind="primary"] {background:linear-gradient(90deg,#dc2626,#f97316);border:0;color:white;font-weight:700;border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def hero(eyebrow, title, subtitle):
    st.markdown(
        f'<div class="fs-hero"><div class="fs-eyebrow">{eyebrow}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def asset(name):
    path = ASSET_DIR / name
    return path if path.exists() else None


def find_dataset_root():
    """Find the dataset root using the expected train/valid/test image folders."""
    candidates = [
        DATASET_DIR,
        project_path("Indoor-Fire-Smoke"),
        project_path("Indoor_Fire_Smoke"),
        project_path("data"),
        PROJECT_DIR,
    ]
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        if any((resolved / split / "images").is_dir() for split in ("train", "valid", "test", "validation")):
            return resolved
    return None


def read_image_labels(label_path):
    """Read class IDs from YOLO TXT rows; unknown/missing labels stay unknown."""
    if not label_path.exists():
        return None
    class_ids = set()
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.strip().split()
            if fields:
                class_ids.add(int(fields[0]))
    except (OSError, ValueError):
        return None
    names = []
    if 0 in class_ids:
        names.append("Fire")
    if 1 in class_ids:
        names.append("Smoke")
    return names


@st.cache_resource(show_spinner="Loading the fine-tuned Vision Transformer...")
def load_model():
    if MODEL_ID:
        model_source = MODEL_ID
    elif LOCAL_MODEL_DIR.exists():
        model_source = str(LOCAL_MODEL_DIR)
    else:
        raise FileNotFoundError(
            "Model not found. Set HF_MODEL_ID to your Hugging Face model repository "
            "or place the exported model files in ./best_model."
        )
    processor = ViTImageProcessor.from_pretrained(model_source)
    model = ViTForImageClassification.from_pretrained(model_source)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    label_file = Path(model_source) / "class_labels.json"
    labels = ["Fire", "Smoke"]
    threshold = DEFAULT_THRESHOLD
    if label_file.exists():
        with label_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        labels = metadata.get("class_names", labels)
        threshold = float(metadata.get("threshold", threshold))
    return model, processor, device, labels, threshold


def predict_image(image):
    model, processor, device, class_names, threshold = load_model()
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.sigmoid(logits)[0].detach().cpu().tolist()
    results = {
        class_names[i] if i < len(class_names) else f"Class {i}": float(probabilities[i])
        for i in range(min(len(probabilities), len(class_names)))
    }
    predicted = [name for name, score in results.items() if score >= threshold]
    return results, predicted, threshold, device


def metric_card(label, value, note):
    st.markdown(
        f'<div class="fs-card"><div class="fs-kpi-label">{label}</div>'
        f'<div class="fs-kpi-value">{value}</div><div class="fs-kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# Navigation
# -------------------------------------------------------------------
st.sidebar.markdown("# 🔥 FireSight AI")
st.sidebar.caption("VISION TRANSFORMER · FIRE & SMOKE")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview Dashboard",
        "Live Image Analysis",
        "Sample Image Gallery",
        "Model Performance",
        "Project & Workflow",
        "Dataset & Augmentation",
        "Use Cases & Impact",
        "About & Limitations",
    ],
    key="firesight_nav",
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model**")
st.sidebar.caption("ViT-B/16 · Multi-label classification")
st.sidebar.caption("Class 0 = Fire · Class 1 = Smoke")
st.sidebar.markdown(
    '<div class="fs-muted">Academic prototype — not a certified fire alarm.</div>',
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Overview
# -------------------------------------------------------------------
if page == "Overview Dashboard":
    hero(
        "AI-POWERED IMAGE CLASSIFICATION",
        "FireSight AI",
        "A fine-tuned Vision Transformer prototype for classifying visible fire and smoke indicators in uploaded images.",
    )
    st.markdown(
        '<div class="fs-callout"><b>Project objective</b><br>'
        'Explore image-level multi-label classification with ViT-B/16 and present its predictions '
        'and measured test performance in an accessible interface.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    cols = st.columns(4)
    with cols[0]:
        metric_card("Architecture", "ViT-B/16", "Fine-tuned pretrained checkpoint")
    with cols[1]:
        metric_card("Test subset accuracy", "96.67%", "Both labels must match")
    with cols[2]:
        metric_card("Test macro F1", "98.39%", "Fire and Smoke weighted equally")
    with cols[3]:
        metric_card("Dataset", "5,000 images", "3,500 train · 750 validation · 750 test")

    st.write("")
    left, right = st.columns([1.2, .8], gap="large")
    with left:
        st.subheader("From image to prediction")
        st.markdown(
            '<div class="fs-card"><span class="fs-pill">1 · Input image</span> → '
            '<span class="fs-pill">2 · Resize & normalize</span> → '
            '<span class="fs-pill">3 · ViT-B/16</span> → '
            '<span class="fs-pill">4 · Fire / Smoke scores</span>'
            '<p style="margin-top:1rem">The model outputs two independent scores. At threshold 0.50, '
            'each label is evaluated separately, so an image can be classified as Fire, Smoke, both, or neither.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("Quick actions")
        if st.button("🔥 Analyze an image", type="primary", use_container_width=True):
            st.session_state["firesight_page"] = "Live Image Analysis"
            st.rerun()
        st.write("")
        st.markdown(
            '<div class="fs-card"><b>Explore the research</b><p class="fs-muted">'
            'Review the model workflow, data augmentation, evaluation results, use cases, and limitations '
            'using the sidebar.</p></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    st.warning(
        "This app analyses uploaded images only. It does not monitor live cameras, locate objects, "
        "confirm an emergency, or replace certified fire-safety equipment."
    )

# -------------------------------------------------------------------
# Live image analysis
# -------------------------------------------------------------------
elif page == "Live Image Analysis":
    hero(
        "INTERACTIVE DEMONSTRATION",
        "Live Image Analysis",
        "Upload an image and explicitly run the fine-tuned model to inspect Fire and Smoke scores.",
    )
    uploaded = st.file_uploader(
        "Drag and drop an image or browse files",
        type=["jpg", "jpeg", "png", "webp"],
        help="One image per analysis. Supported formats: JPG, JPEG, PNG and WEBP.",
        key="firesight_upload",
    )
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
        except Exception:
            st.error("This file could not be opened as an image. Please choose another image.")
        else:
            input_col, result_col = st.columns([1, 1], gap="large")
            with input_col:
                st.subheader("Input image")
                st.image(image, use_container_width=True)
                st.caption(f"File: {uploaded.name} · Dimensions: {image.width} × {image.height}")
                analyse = st.button("Run image analysis", type="primary", use_container_width=True)
            with result_col:
                st.subheader("Prediction results")
                if analyse:
                    try:
                        with st.spinner("Loading model and analysing image..."):
                            results, predicted, threshold, device = predict_image(image)
                        st.session_state["firesight_last_result"] = {
                            "filename": uploaded.name,
                            "results": results,
                            "predicted": predicted,
                            "threshold": threshold,
                            "device": device.type.upper(),
                            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        }
                        entry = st.session_state["firesight_last_result"]
                        if len(predicted) == 2:
                            st.success("Model prediction: Fire + Smoke")
                        elif len(predicted) == 1:
                            st.success(f"Model prediction: {predicted[0]}")
                        else:
                            st.warning("Neither label crossed the decision threshold")
                        st.caption(f"Threshold: {threshold:.2f} · Device: {device.type.upper()}")
                        for label, score in results.items():
                            st.write(f"**{label}** · {score:.1%}")
                            st.progress(min(max(score, 0.0), 1.0))
                        st.info(
                            "These are model scores, not calibrated estimates of real-world risk. "
                            "A low score does not prove that an image or location is safe."
                        )
                    except Exception as exc:
                        st.error("The model could not be loaded or the image could not be analysed.")
                        st.code(str(exc))
                        st.info(
                            "For local use, ensure ./best_model contains the exported model files. "
                            "For a hosted model, set HF_MODEL_ID to the Hugging Face repository ID."
                        )
                else:
                    st.markdown(
                        '<div class="fs-card"><b>Ready to analyse</b><p class="fs-muted">'
                        'Select “Run image analysis” after choosing an image. Results will appear here.</p></div>',
                        unsafe_allow_html=True,
                    )

    last = st.session_state.get("firesight_last_result")
    if last:
        st.subheader("Download last prediction report")
        report = {
            "application": "FireSight AI",
            "filename": last["filename"],
            "timestamp_utc": last["timestamp"],
            "predicted_labels": last["predicted"],
            "probabilities": last["results"],
            "decision_threshold": last["threshold"],
            "device": last["device"],
            "note": "Academic model output; not a certified safety assessment.",
        }
        st.download_button(
            "Download prediction report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="firesight_prediction_report.json",
            mime="application/json",
        )
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["filename", "timestamp_utc", "label", "probability", "threshold", "device"])
        for label, score in last["results"].items():
            writer.writerow([last["filename"], last["timestamp"], label, f"{score:.6f}", last["threshold"], last["device"]])
        st.download_button(
            "Download prediction report (CSV)",
            data=csv_buffer.getvalue(),
            file_name="firesight_prediction_report.csv",
            mime="text/csv",
        )

# -------------------------------------------------------------------
# Sample gallery: notebook-produced test predictions, not invented samples
# -------------------------------------------------------------------
elif page == "Sample Image Gallery":
    hero(
        "NOTEBOOK EXAMPLES",
        "Sample Image Gallery",
        "Examples and visual outputs saved from the project notebook.",
    )
    st.write(
        "The image below is the notebook's saved test-prediction visualisation. Each example includes "
        "its actual labels, predicted labels, and Fire/Smoke scores from the evaluation run."
    )
    sample_path = asset("test_prediction_examples.png")
    if sample_path:
        st.image(str(sample_path), caption="Saved test-set prediction examples from the Colab notebook", use_container_width=True)
    else:
        st.info(
            "The saved gallery image is not present. Keep the supplied assets folder beside app.py "
            "to display notebook-generated examples."
        )
    st.caption(
        "These are a small set of examples, not a substitute for the full test evaluation. "
        "Do not treat a sample prediction as proof of performance on all environments."
    )

# -------------------------------------------------------------------
# Model performance
# -------------------------------------------------------------------
elif page == "Model Performance":
    hero(
        "VERIFIED EXPERIMENT OUTPUT",
        "Model Performance",
        "Test metrics and confusion matrices recorded in the Colab notebook at threshold 0.50.",
    )
    k1, k2, k3 = st.columns(3)
    with k1:
        metric_card("Subset accuracy", "96.67%", "Both labels must match for an image")
    with k2:
        metric_card("Macro F1-score", "98.39%", "Average of class-wise F1 scores")
    with k3:
        metric_card("Evaluation threshold", "0.50", "Applied independently to each label")

    st.subheader("Per-class classification report")
    st.dataframe(
        [
            {"Class": "Fire", "Precision": "0.99", "Recall": "1.00", "F1-score": "1.00", "Positive support": 392},
            {"Class": "Smoke", "Precision": "0.98", "Recall": "0.97", "F1-score": "0.97", "Positive support": 424},
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Per-class precision, recall and F1 are displayed as rounded in the notebook output. "
        "The macro F1 and subset accuracy use the notebook's saved values."
    )

    cm_path = asset("confusion_matrices.png")
    if cm_path:
        st.subheader("Confusion matrices")
        st.image(str(cm_path), caption="Original confusion-matrix figure extracted from the saved Colab notebook.", use_container_width=True)
    else:
        st.warning("Confusion-matrix image missing from assets/confusion_matrices.png.")

    st.subheader("Counts from the confusion matrices")
    cm_cols = st.columns(2)
    for col, label in zip(cm_cols, ["Fire", "Smoke"]):
        cm = VERIFIED_RESULTS["confusion_matrices"][label]
        with col:
            st.markdown(f"**{label}**")
            st.dataframe(
                [
                    {"Actual label": "Absent", "Predicted absent": cm["TN"], "Predicted present": cm["FP"]},
                    {"Actual label": "Present", "Predicted absent": cm["FN"], "Predicted present": cm["TP"]},
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.write(
                f"False positives: **{cm['FP']}** · False negatives: **{cm['FN']}**"
            )

    st.subheader("How to interpret errors")
    st.markdown(
        """
        - **False positive:** the model predicts a label as present when the ground-truth label is absent.
        - **False negative:** the model predicts a label as absent when the ground-truth label is present.
        - **Fire:** the shown matrix contains 3 false positives and 0 false negatives.
        - **Smoke:** the shown matrix contains 10 false positives and 14 false negatives.
        """
    )
    st.info(
        "Support-count clarification: the classification report's support (Fire 392, Smoke 424) counts "
        "positive ground-truth instances for each label. Each one-vs-rest confusion matrix also counts "
        "negative instances, so each matrix totals 750 test images. These figures are compatible; "
        "the different totals are not, by themselves, a discrepancy."
    )

    st.subheader("Training history")
    curve1, curve2 = st.columns(2)
    with curve1:
        p = asset("training_loss.png")
        if p:
            st.image(str(p), caption="Training and validation loss from notebook output", use_container_width=True)
        else:
            st.caption("Training-loss plot unavailable. Restore assets/training_loss.png.")
    with curve2:
        p = asset("validation_macro_f1.png")
        if p:
            st.image(str(p), caption="Validation macro-F1 from notebook output", use_container_width=True)
        else:
            st.caption("Validation metric plot unavailable. Restore assets/validation_macro_f1.png.")

# -------------------------------------------------------------------
# Project and interactive workflow
# -------------------------------------------------------------------
elif page == "Project & Workflow":
    hero(
        "RESEARCH OVERVIEW",
        "Project & ViT Workflow",
        "How the labelled image dataset becomes two independent image-level predictions.",
    )
    st.subheader("Research problem")
    st.write(
        "This project investigates whether a pretrained Vision Transformer can classify images for visible "
        "fire and smoke indicators. The output is image-level classification, not object detection."
    )
    st.subheader("Objectives")
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="fs-card"><h3>01 · Prepare</h3><p>Validate YOLO annotation files and convert object annotations into image-level Fire/Smoke labels.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="fs-card"><h3>02 · Fine-tune</h3><p>Fine-tune pretrained ViT-B/16 with two sigmoid outputs and a multi-label loss.</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="fs-card"><h3>03 · Evaluate</h3><p>Use a held-out test split and inspect subset accuracy, class-wise metrics and confusion matrices.</p></div>', unsafe_allow_html=True)

    st.subheader("Interactive model pipeline")
    stages = [
        ("1. Input image", "RGB image uploaded by the user."),
        ("2. Preprocessing", "Resize to 224 × 224 and apply the pretrained checkpoint's normalization."),
        ("3. Patch embeddings", "ViT-B/16 represents the image as 16 × 16 pixel patches and embeds them as tokens."),
        ("4. Transformer encoder", "Self-attention layers model relationships between image patch tokens."),
        ("5. Classification head", "Two logits correspond to Fire and Smoke."),
        ("6. Sigmoid + threshold", "Independent scores are compared with threshold 0.50."),
    ]
    selected_stage = st.select_slider(
        "Explore the workflow step by step",
        options=[s[0] for s in stages],
        value=stages[0][0],
    )
    idx = next(i for i, s in enumerate(stages) if s[0] == selected_stage)
    st.markdown(
        f'<div class="fs-card"><div class="fs-eyebrow">PIPELINE STAGE {idx+1} OF {len(stages)}</div>'
        f'<h2>{stages[idx][0]}</h2><p>{stages[idx][1]}</p></div>',
        unsafe_allow_html=True,
    )
    st.progress((idx + 1) / len(stages))
    st.caption(
        "Architecture description follows the ViT-B/16 checkpoint and the notebook inference pipeline. "
        "The app does not visualise attention maps or object locations."
    )

# -------------------------------------------------------------------
# Dataset and augmentation
# -------------------------------------------------------------------
elif page == "Dataset & Augmentation":
    hero(
        "DATA TRANSPARENCY",
        "Dataset & Augmentation",
        "Browse actual dataset images or the bundled sample gallery, and compare the training augmentation preview.",
    )
    d = VERIFIED_RESULTS["dataset"]
    cols = st.columns(4)
    with cols[0]:
        metric_card("Total images", f'{d["total"]:,}', "Original dataset")
    with cols[1]:
        metric_card("Training", f'{d["train"]:,}', "70% of images")
    with cols[2]:
        metric_card("Validation", f'{d["validation"]:,}', "15% of images")
    with cols[3]:
        metric_card("Test", f'{d["test"]:,}', "15% of images")

    st.subheader("Class mapping")
    x, y = st.columns(2)
    with x:
        st.markdown('<div class="fs-card"><h3>Class 0 · Fire</h3><p>Images with at least one Fire annotation.</p></div>', unsafe_allow_html=True)
    with y:
        st.markdown('<div class="fs-card"><h3>Class 1 · Smoke</h3><p>Images with at least one Smoke annotation. An image may contain both classes.</p></div>', unsafe_allow_html=True)

    st.subheader("Explore actual dataset images")
    st.caption("Images are loaded from the full local dataset when available, or from the bundled real-image samples for cloud deployment. Labels are read from their matching original YOLO TXT files; no stock images are used.")
    dataset_root = find_dataset_root()
    using_sample_gallery = dataset_root is None
    if using_sample_gallery:
        image_dir = project_path("gallery_samples/images")
        label_dir = project_path("gallery_samples/labels")
        if image_dir.is_dir():
            st.markdown(
                "Showing selected real images from the [Indoor Fire Smoke Dataset on Zenodo]"
                "(https://zenodo.org/records/15826133). Captions use the paired source annotation files."
            )
        else:
            st.warning(
                "The full dataset and bundled gallery samples are unavailable. Download the dataset from "
                "https://zenodo.org/records/15826133 and place its train/valid/test image and label folders "
                "under dataset/, or restore gallery_samples/ for the compact cloud gallery."
            )
    else:
        split_choice = st.selectbox("Dataset split", ["train", "valid", "test"], index=0)
        split_dir = dataset_root / split_choice
        image_dir = split_dir / "images"
        label_dir = split_dir / "labels"

    if image_dir.is_dir():
        if not label_dir.is_dir():
            st.warning(f"Label folder not found: {label_dir}")
        image_paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        filter_choice = st.selectbox("Filter by annotation", ["All annotated images", "Fire", "Smoke", "Fire + Smoke", "Missing label file"], index=0)
        gallery_items = []
        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            labels = read_image_labels(label_path)
            if filter_choice == "Missing label file":
                include = not label_path.exists()
            elif filter_choice == "Fire":
                include = labels is not None and "Fire" in labels
            elif filter_choice == "Smoke":
                include = labels is not None and "Smoke" in labels
            elif filter_choice == "Fire + Smoke":
                include = labels is not None and "Fire" in labels and "Smoke" in labels
            else:
                include = labels is not None
            if include:
                gallery_items.append((image_path, labels, label_path.exists()))

        if using_sample_gallery:
            st.caption(f"Bundled gallery · Showing {len(gallery_items):,} matching sample image(s).")
        else:
            st.caption(f"Dataset root: {dataset_root} · Showing {len(gallery_items):,} matching image(s).")
        if not gallery_items:
            st.info("No images match this filter. Check the split's labels folder and annotation files.")
        else:
            display_count = st.slider("Images to display", min_value=4, max_value=min(24, max(4, len(gallery_items))), value=min(12, max(4, len(gallery_items))), step=4)
            gallery_items = gallery_items[:display_count]
            for offset in range(0, len(gallery_items), 4):
                row = st.columns(4)
                for column, item in zip(row, gallery_items[offset:offset + 4]):
                    image_path, labels, has_label_file = item
                    with column:
                        try:
                            with Image.open(image_path) as im:
                                st.image(im.convert("RGB"), use_container_width=True)
                        except Exception:
                            st.error(f"Could not open {image_path.name}")
                            continue
                        st.caption(image_path.name)
                        if labels is None:
                            st.warning("Label unreadable or missing")
                        elif labels:
                            st.markdown(" · ".join(f"`{label}`" for label in labels))
                        else:
                            st.caption("No class IDs in label file")
    elif not using_sample_gallery:
        st.warning(f"Image folder not found: {image_dir}")

    st.subheader("Preprocessing and augmentation")
    st.markdown(
        """
        **Preprocessing:** RGB input, resize to 224 × 224 pixels, then use the selected pretrained ViT image processor.

        **Training only:** random augmentation is applied dynamically while training. Validation and test images use deterministic preprocessing without random augmentation.
        """
    )
    st.info(
        "Augmentation is performed on the fly. It does not create extra permanent files, so the stored dataset remains "
        "5,000 original images."
    )
    aug_path = asset("augmentation_example.png")
    if aug_path:
        st.subheader("Original vs augmented training image")
        st.image(str(aug_path), caption="Augmentation preview exported from the training notebook.", use_container_width=True)
    else:
        st.caption("To display the notebook's before-and-after augmentation example, place augmentation_example.png in the assets folder.")

# -------------------------------------------------------------------
# Use cases and impact
# -------------------------------------------------------------------
elif page == "Use Cases & Impact":
    hero(
        "POTENTIAL APPLICATIONS",
        "Use Cases & Impact",
        "Research directions for image-based fire and smoke classification.",
    )
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown('<div class="fs-card"><h3>🏭 Industrial facilities</h3><p>Explore classification of safety-related images from warehouses, workshops and industrial scenes.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="fs-card"><h3>🏢 Indoor environments</h3><p>Study how lighting, occlusion, image quality and scene variation affect classification.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="fs-card"><h3>👷 Human-assisted review</h3><p>Investigate whether predictions could help prioritise images for review by a trained person.</p></div>', unsafe_allow_html=True)

    st.subheader("Potential impact")
    st.write(
        "The project demonstrates transfer learning for a fire-safety-related computer-vision task and provides "
        "a basis for examining model performance on real imagery. Any future operational use would require "
        "representative testing, system-level validation, human oversight and safety engineering."
    )
    st.subheader("Future work")
    st.markdown(
        """
        - Test generalisation on diverse, independently collected indoor scenes.
        - Review errors and choose thresholds using validation data.
        - Add more detailed robustness and calibration evaluation.
        - Explore real-time video integration as a separate future project, with appropriate safety controls.
        - Investigate explainability tools, while clearly distinguishing them from verified causal explanations.
        """
    )

# -------------------------------------------------------------------
# About and limitations
# -------------------------------------------------------------------
elif page == "About & Limitations":
    hero(
        "PROJECT INFORMATION",
        "About FireSight AI",
        "An academic demonstration of ViT-B/16 multi-label image classification.",
    )
    st.subheader("Technology stack")
    tech = st.columns(4)
    for col, (name, value) in zip(
        tech,
        [("Architecture", "ViT-B/16"), ("Framework", "PyTorch"), ("Model library", "Hugging Face Transformers"), ("Interface", "Streamlit")],
    ):
        with col:
            metric_card(name, value, "Used in the prototype")

    st.subheader("Limitations and responsible use")
    st.markdown(
        """
        - This app analyses one uploaded image at a time; continuous camera monitoring is not implemented.
        - It classifies the whole image and does not return bounding boxes or object locations.
        - Dataset bias, unfamiliar scenes, lighting, occlusion and image quality may affect predictions.
        - Model scores are not calibrated guarantees of actual fire risk.
        - A negative prediction does not prove safety.
        - This prototype is not certified for emergency response and must not replace approved fire alarms, sensors or safety procedures.
        """
    )
    st.subheader("Configuration")
    st.code(
        "Local model: ./best_model\n"
        "Optional hosted model: HF_MODEL_ID=<your Hugging Face repository ID>\n"
        "Optional threshold: PREDICTION_THRESHOLD=0.5",
        language="text",
    )

st.markdown("---")
st.markdown(
    '<div class="fs-muted">FireSight AI · ViT-B/16 · Academic prototype · '
    'Image-level classification only — not a certified fire detection system.</div>',
    unsafe_allow_html=True,
)
