"""About: how it works, what is ready, real model results, and honest limits."""
from __future__ import annotations

import streamlit as st

from config import settings
from modules.freshness_classifier import load_checkpoint_metrics
from ui import theme
from ui.state import get_classifier, get_detector, get_ocr_engine

FLOW = """
digraph G {
  rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.35;
  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#C7DAE2", fontname="Helvetica", fontsize=11, fontcolor="#12302A"];
  edge [color="#4B625C"];
  photo [label="Food photo"]; yolo [label="YOLOv8 produce +\\nlocal YOLO-World foods"];
  live [label="Live camera\\nconfigured detector"]; crop [label="Crop supported produce"];
  mobile [label="MobileNetV2\\nfresh / aging / spoiled"]; label [label="Package photo or\\nlive camera"]; ocr [label="OCR reads\\nthe label and date"];
  confirm [label="You confirm\\nfood and dates"]; db [label="SQLite fridge", fillcolor="#DDF0E5"];
  fuzzy [label="Fuzzy logic\\nscores priority", fillcolor="#FDF0D2"]; plan [label="Use first / check list", fillcolor="#FDF0D2"];
  photo -> yolo -> crop -> mobile -> confirm -> db; live -> yolo; label -> ocr -> confirm; db -> fuzzy -> plan;
}
"""


def render() -> None:
    theme.hero("How FreshTrack works", "An agent that looks at your food, reads its label, remembers what you have, and decides what to eat first.")
    st.graphviz_chart(FLOW, use_container_width=True)

    theme.section("What is ready")
    detector_model, freshness_model, ocr_model = get_detector(), get_classifier(), get_ocr_engine()
    detector, classifier, ocr = detector_model.status(), freshness_model.status(), ocr_model.status()
    rows = [
        ("Food detection (YOLO)", detector),
        ("Freshness model (MobileNetV2)", classifier),
        ("Date reader (Tesseract OCR)", ocr),
    ]
    html = "".join(f'<div class="row"><span>{theme.esc(n)}</span><span>{"" if s.available else "Not ready. "}{theme.esc(s.message)}</span></div>' for n, s in rows)
    st.markdown(f'<div class="plain-list">{html}</div>', unsafe_allow_html=True)
    try:
        supported_foods = ", ".join(freshness_model.supported_foods()) or "none"
        theme.note(f"Foods supported by the loaded freshness checkpoint: {supported_foods}. Detection support alone does not provide a freshness result.", "info")
    except RuntimeError as exc:
        st.error(f"Could not read the freshness model's supported-food list: {exc}")

    metrics = load_checkpoint_metrics(freshness_model.checkpoint)
    if metrics:
        theme.section("Freshness model results", "Measured on photos the model never saw during training.")
        test = metrics["test"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{test['accuracy'] * 100:.1f}%")
        c2.metric("Macro F1", f"{test['macro_f1']:.3f}")
        c3.metric("Test images", f"{test['n']:,}")
        table = [
            {"Class": settings.FRESHNESS_LABELS.get(c, c), "Precision": round(v["precision"], 3), "Recall": round(v["recall"], 3),
             "F1": round(v["f1"], 3), "Images": v["support"]}
            for c, v in test["per_class"].items()
        ]
        st.dataframe(table, hide_index=True, use_container_width=True)
        if metrics.get("note"):
            theme.note(metrics["note"], "info")
        if "real_world" in metrics:
            st.metric("Accuracy on your own photos", f"{metrics['real_world']['accuracy'] * 100:.1f}%")
        theme.note("These photos come from a public research dataset. Pictures from your own fridge, with different light and angles, will usually score lower.", "info")
    else:
        theme.section("Freshness model results")
        theme.note("No results yet. Train and evaluate the model (see documentation/MODEL_TRAINING_GUIDE.md) and the real numbers will appear here.", "info")

    theme.section("What it cannot do")
    st.markdown(
        "- **It judges looks, not safety.** A fresh-looking fruit can still be unsafe, and FreshTrack cannot see bacteria or toxins.\n"
        "- **Freshness only covers foods listed above.** Detection support does not provide freshness estimates for other foods.\n"
        "- **Food detection has separate paths.** Photo scans combine the 63-class produce detector with a local open-vocabulary model prompted for additional foods; the latter is not trained on FreshTrack meat/dairy data and its suggestions need confirmation. Live camera uses the configured primary detector. Optional Roboflow cloud inference sends images only when enabled.\n"
        "- **Date reading can miss or misread.** Curved, shiny or faded labels fail, and day/month order is never guessed.\n"
        "- **The priority score is a ranking, not a verdict.** A low score never means a food is safe to eat."
    )
