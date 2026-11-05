"""Scan: photograph food to add it, or photograph a package to read its date."""
from __future__ import annotations

import hashlib
from datetime import date, timedelta
from typing import Optional

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from config import settings
from modules import image_processor as ip
from modules.expiration_date_reader import DateCandidate, read_expiration_dates
from modules.food_detector import Detection
from modules.freshness_classifier import FreshnessPrediction
from modules.live_camera import LiveFoodProcessor
from modules.roboflow_workflow import merge_detections
from ui import theme
from ui.state import get_classifier, get_detector, get_inventory, get_ocr_engine, get_roboflow_detector, goto

MODES = ["Food photo", "Package date"]
SAFETY = "Freshness here is about how the food looks. It cannot tell you whether it is safe to eat."
MANUAL_PACKAGE_DATE_TYPES = ("expiration", "best_before")


def render() -> None:
    theme.hero("Scan", "Photograph food to add it to your fridge, or photograph a package to read its date.")
    st.session_state.setdefault("scan_mode", MODES[0])
    with st.container(key="modepick"):
        st.radio("What are you scanning?", MODES, key="scan_mode", horizontal=True, label_visibility="collapsed")
    st.write("")
    if st.session_state["scan_mode"] == MODES[0]:
        _food()
    else:
        _dates()


def _image_input(key: str) -> Optional[bytes]:
    upload_tab, camera_tab = st.tabs(["Upload a photo", "Use the camera"])
    with upload_tab:
        uploaded = st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp"], key=f"{key}_up", label_visibility="collapsed")
    with camera_tab:
        shot = st.camera_input("Camera", key=f"{key}_cam", label_visibility="collapsed")
    chosen = shot or uploaded
    return chosen.getvalue() if chosen else None


# ----------------------------------------------------------------------------
# Food photo
# ----------------------------------------------------------------------------
def _analyze(image_bgr, classifier, classifier_ready: bool, roboflow=None) -> dict:
    local_error = None
    extended_error = None
    detector = get_detector()
    try:
        detections = detector.detect(image_bgr)
    except Exception as exc:  # noqa: BLE001 - show any model problem in the UI
        local_error = str(exc)
        detections = []
    try:
        detections = merge_detections(detections, detector.detect_extended(image_bgr))
    except Exception as exc:  # noqa: BLE001 - preserve local detections and report extension failure
        extended_error = str(exc)
    remote_error = None
    if roboflow is not None:
        try:
            detections = merge_detections(detections, roboflow.detect(image_bgr))
        except Exception as exc:  # noqa: BLE001 - preserve local results and show cloud failure
            remote_error = str(exc)
    predictions = []
    freshness_error = None
    for det in detections:
        prediction = None
        if classifier_ready and det.category == "Fruits & Vegetables":
            try:
                if not classifier.supports_food(det.label):
                    predictions.append(None)
                    continue
                prediction = classifier.predict(ip.bgr_to_rgb(ip.crop_box(image_bgr, det.box)))
            except Exception as exc:  # noqa: BLE001 - display assessment failures instead of hiding them
                freshness_error = str(exc)
        predictions.append(prediction)
    return dict(
        error=local_error,
        extended_error=extended_error,
        remote_error=remote_error,
        freshness_error=freshness_error,
        detections=detections,
        predictions=predictions,
        annotated=ip.draw_detections(image_bgr, detections),
    )


def _freshness_block(prediction) -> None:
    if prediction is None:
        return
    pct = f"{prediction.confidence * 100:.0f}%"
    if prediction.uncertain:
        theme.note(f"Not sure how fresh this is. Best guess: {prediction.display.lower()} ({pct}). Try a closer, clearer photo.", "info")
    elif prediction.label == "spoiled":
        theme.note(f"This looks spoiled ({pct} sure). Do not eat it.", "bad")
    elif prediction.label == "aging":
        theme.note(f"Looks aging or overripe ({pct} sure). Use it soon.")
    else:
        theme.note(f"Looks fresh ({pct} sure).", "info")


def _food() -> None:
    inv, classifier = get_inventory(), get_classifier()
    state = classifier.status()
    scan_mode = st.radio(
        "Food scan mode",
        ["Live camera", "Photo or upload"],
        horizontal=True,
        key="food_scan_mode",
    )
    if scan_mode == "Live camera":
        _live_food_camera(inv)
        return

    roboflow = _configured_roboflow()
    send_to_roboflow = False
    if roboflow is not None:
        send_to_roboflow = st.checkbox(
            "Use Roboflow cloud detection for this photo",
            value=False,
            help="When selected, this photo is sent to your configured Roboflow Workflow. "
                 "Usage limits or charges may apply.",
        )
        st.caption("Roboflow is optional; the local detectors run either way.")
    else:
        st.caption("Cloud detection is off; photos are scanned locally. No Roboflow account is needed.")
    st.caption(
        "A 25 MB local model helps recognize meat, dairy, and other foods; it downloads once if missing. "
        "Photos are not sent to its provider."
    )

    data = _image_input("food")
    if not state.available:
        theme.note(f"Freshness check is off. {state.message}", "info")
    if data is None:
        st.caption(
            "Photo scans use the built-in produce detector and a broader local model for meat, dairy, and other foods. "
            "The broader model downloads on first use."
        )
        _manual_form(inv, classifier, state.available, None, "none")
        return

    try:
        image = ip.load_image(data)
    except Exception:  # noqa: BLE001
        st.error("That file could not be read as an image. Try a JPG or PNG.")
        return
    digest = hashlib.sha256(data).hexdigest()
    result_key = f"{digest}_{'cloud' if send_to_roboflow else 'local'}_extended"
    cache = st.session_state.setdefault("food_cache", {})
    if result_key not in cache:
        with st.spinner("Looking for food..."):
            cache[result_key] = _analyze(
                image, classifier, state.available, roboflow if send_to_roboflow else None
            )
    result = cache[result_key]

    left, right = st.columns([1, 1], gap="large")
    if result["annotated"] is not None:
        left.image(result["annotated"], use_container_width=True)
    else:
        left.image(ip.bgr_to_rgb(image), use_container_width=True)

    with right:
        if result.get("extended_error"):
            st.warning(f"Extra local food detection failed: {result['extended_error']}")
        if result.get("remote_error"):
            st.warning(f"Roboflow cloud inference failed: {result['remote_error']}")
        if result.get("freshness_error"):
            st.warning(f"Freshness assessment failed: {result['freshness_error']}")
        if result["error"]:
            if result["detections"]:
                st.warning(f"The built-in detector failed, but another detector returned results: {result['error']}")
            else:
                st.error(f"Detection could not run: {result['error']}")
        if not result["detections"]:
            theme.note("No supported food was detected by the local models. Try a closer, well-lit photo or add it by hand below.", "info")
        else:
            _detection_forms(inv, result_key, result, state.available, image)
    st.write("")
    _manual_form(inv, classifier, state.available, image, result_key, expanded=not result["detections"])


def _configured_roboflow():
    try:
        return get_roboflow_detector()
    except ValueError as exc:
        st.error(f"Roboflow configuration is invalid: {exc}")
        return None


def _live_food_camera(inv) -> None:
    detector, classifier, ocr_engine = get_detector(), get_classifier(), get_ocr_engine()
    roboflow = _configured_roboflow()
    use_roboflow = False
    if roboflow is not None:
        use_roboflow = st.checkbox(
            "Send selected live camera frames to Roboflow for extra food detections",
            value=False,
            help="Only selected frames are sent, about once every 3 seconds while a request is available.",
        )
        if use_roboflow:
            st.warning(
                "The live camera sends a frame to Roboflow about every 3 seconds while running. "
                "This uses a cloud service and may incur usage charges. Stop the camera to stop sending frames."
            )
    else:
        st.caption("Roboflow live detection is off. To enable it, configure ROBOFLOW_API_KEY in your local .env and restart FreshTrack.")
    freshness_status = classifier.status()
    if detector.weights.name == "yolov8n.pt":
        st.caption("Active detector: COCO YOLOv8. It has no cucumber class, so it may label cucumber as carrot.")
    else:
        st.caption(f"Active live food detector: {detector.status().message}")
    st.caption("The broader meat and dairy detector is used for photo scans; the live camera remains produce-focused.")
    if not freshness_status.available:
        theme.note(f"Live freshness assessment is off. {freshness_status.message}", "info")

    camera_column, result_column = st.columns([1.2, 1], gap="medium")
    with camera_column:
        st.caption("Start the camera and point it at food. Detection runs continuously.")
        with st.container(key="live-food-camera"):
            ctx = webrtc_streamer(
                key="freshtrack-live-food",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": []},
                frontend_rtc_configuration={"iceServers": []},
                video_processor_factory=lambda: LiveFoodProcessor(
                    detector,
                    classifier,
                    ocr_engine,
                    freshness_status.available,
                    False,
                    roboflow if use_roboflow else None,
                ),
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 1280},
                        "height": {"ideal": 720},
                        "frameRate": {"ideal": 15, "max": 30},
                    },
                    "audio": False,
                },
                async_processing=True,
                video_html_attrs={
                    "autoPlay": True,
                    "controls": True,
                    "muted": True,
                    "style": {"width": "100%", "height": "auto", "objectFit": "contain"},
                },
            )
    with result_column:
        with st.container(border=True):
            _live_food_review(ctx, inv)

    st.caption(
        "Green boxes show live food detections. Freshness is an appearance estimate, not a food-safety test."
    )


@st.fragment(run_every="1s")
def _live_food_review(ctx, inv) -> None:
    processor = ctx.video_processor
    if processor is None:
        theme.section("Live food recognition", "Start the camera to see detected foods and confirm an item.")
        return
    _, observations = processor.latest_observation()
    errors = processor.latest_errors()
    if errors:
        for error in errors:
            st.error(error)

    if not observations:
        st.session_state.pop("live_food_saved_signature", None)
        theme.section("Live food recognition")
        st.caption("No food detected yet. Move a food item into the camera view and use good lighting.")
        return

    signature = "|".join(sorted({item.label.lower() for item, _ in observations}))
    if signature == st.session_state.get("live_food_saved_signature"):
        st.success("The detected item was added. This result will stay until another food is detected.")
        return
    st.session_state.pop("live_food_saved_signature", None)

    theme.section("Live food recognition", "Detections update continuously. Select which detected food to confirm.")
    if len(observations) > 1:
        selected = st.selectbox(
            "Food to confirm",
            range(len(observations)),
            format_func=lambda index: (
                f"{observations[index][0].label.title()} #{index + 1} "
                f"· {observations[index][0].confidence:.0%}"
            ),
            key=f"live_food_choice_{signature}",
        )
    else:
        selected = 0
    detection, prediction = observations[selected]
    candidate = detection.label.title()
    category = detection.category
    st.success(
        f"Live camera detected: {candidate} · food-detection confidence {detection.confidence:.0%}."
    )
    if detection.source == "roboflow":
        st.caption("Detection source: Roboflow cloud")
    if detection.quality:
        quality_text = "visually labeled fresh" if detection.quality == "fresh" else "visibly defective in the training dataset"
        st.caption(f"Dataset appearance label: {quality_text}. This is not a food-safety assessment.")
    freshness_info = _live_freshness_info(detection, prediction)
    if freshness_info:
        st.info(freshness_info)

    form_key = f"live_food_confirm_{abs(hash(signature + candidate))}"
    with st.form(form_key):
        name = st.text_input("Confirm or edit item name", value=candidate, key=f"{form_key}_name")
        selected_category = st.selectbox(
            "Category",
            settings.FOOD_CATEGORIES,
            index=settings.FOOD_CATEGORIES.index(category),
            key=f"{form_key}_category",
        )
        c1, c2 = st.columns(2)
        quantity = c1.number_input("How many", min_value=0.0, value=1.0, step=1.0, key=f"{form_key}_quantity")
        unit = c2.selectbox("Unit", settings.UNITS, key=f"{form_key}_unit")
        location = st.selectbox("Where will it go?", settings.STORAGE_LOCATIONS, key=f"{form_key}_location")
        if st.form_submit_button("Confirm and add to fridge", type="primary"):
            freshness_label = prediction.label if prediction and not prediction.uncertain else None
            freshness_confidence = prediction.confidence if freshness_label else None
            try:
                item_id = inv.add_item(
                    name=name,
                    category=selected_category,
                    quantity=quantity,
                    unit=unit,
                    storage_location=location,
                    freshness_label=freshness_label,
                    freshness_confidence=freshness_confidence,
                    image_data=processor.latest_thumbnail(),
                )
                inv.db.add_scan(
                    "detection",
                    f"{name.strip()} confirmed from live camera",
                    {
                        "freshness": freshness_label,
                        "freshness_confidence": freshness_confidence,
                        "suggested_food": candidate,
                        "confirmed_food": name.strip(),
                        "detector_source": detection.source,
                    },
                    item_id,
                    image_data=processor.latest_thumbnail(),
                )
                st.session_state["live_food_saved_signature"] = signature
                st.success(f"Added {name.strip()} to your fridge.")
            except ValueError as exc:
                st.error(str(exc))


def _live_freshness_info(
    detection: Detection,
    prediction: Optional[FreshnessPrediction],
) -> Optional[str]:
    if prediction is not None:
        if prediction.uncertain:
            return (
                f"Freshness appearance: uncertain — best guess {prediction.display.lower()} "
                f"({prediction.confidence:.0%} model confidence). Not a food-safety test."
            )
        return (
            f"Freshness appearance: {prediction.display} "
            f"({prediction.confidence:.0%} model confidence). Not a food-safety test."
        )

    if detection.category != "Fruits & Vegetables":
        return None
    classifier = get_classifier()
    if not classifier.status().available:
        return "Freshness appearance is unavailable because the freshness model is not loaded."
    if not classifier.supports_food(detection.label):
        return (
            f"Freshness appearance is not available for {detection.label}; "
            "this food is not in the loaded freshness model."
        )
    return "Freshness appearance: checking this detected food."


def _group_detected_foods(
    detections: list[Detection],
    predictions: list[Optional[FreshnessPrediction]],
) -> list[tuple[Detection, Optional[FreshnessPrediction], int]]:
    if len(detections) != len(predictions):
        raise ValueError("Each food detection must have a matching freshness result.")
    grouped: dict[tuple[str, str], list[tuple[Detection, Optional[FreshnessPrediction]]]] = {}
    for detection, prediction in zip(detections, predictions):
        key = (detection.label.strip().casefold(), detection.category)
        grouped.setdefault(key, []).append((detection, prediction))

    results = []
    for items in grouped.values():
        detection, prediction = max(items, key=lambda item: item[0].confidence)
        results.append((detection, prediction, len(items)))
    return results


def _detection_forms(inv, digest: str, result: dict, freshness_ready: bool, image) -> None:
    detections, predictions = result["detections"], result["predictions"]
    groups = _group_detected_foods(detections, predictions)
    total_detected = sum(count for _, _, count in groups)
    st.markdown(
        f'<div class="section-sub">Detected {total_detected} item{"s" if total_detected != 1 else ""} '
        f'across {len(groups)} food type{"s" if len(groups) != 1 else ""}. '
        "Repeated detections are combined into quantity.</div>",
        unsafe_allow_html=True,
    )
    picks = []
    for i, (det, pred, count) in enumerate(groups):
        with st.container(border=True):
            st.markdown(f'<div class="found"><div class="big">{i + 1}. {theme.esc(det.label.title())}'
                        f' · {count} detected</div>'
                        f'<div class="small">{det.confidence * 100:.0f}% sure</div></div>', unsafe_allow_html=True)
            if det.source == "open-vocabulary":
                st.caption("Suggested by the broader local model; verify the name before adding.")
            add = st.checkbox("Add to fridge", value=True, key=f"{digest}_{i}_add")
            c1, c2 = st.columns(2)
            name = c1.text_input("Name", value=det.label.title(), key=f"{digest}_{i}_name")
            category = c2.selectbox("Category", settings.FOOD_CATEGORIES, index=settings.FOOD_CATEGORIES.index(det.category), key=f"{digest}_{i}_cat")
            qty = st.number_input("How many", min_value=1.0, value=float(count), step=1.0, key=f"{digest}_{i}_qty")
            if det.quality:
                description = "visually labeled fresh" if det.quality == "fresh" else "visibly defective in the training dataset"
                st.caption(f"Dataset appearance label: {description}. This is not a food-safety assessment.")
            if det.category == "Fruits & Vegetables":
                if pred is None and freshness_ready:
                    st.caption(f"The loaded freshness model does not support {det.label} yet.")
                else:
                    _freshness_block(pred)
            picks.append((add, name, category, qty, pred, det, count))
    where = st.selectbox("Where will they go?", settings.STORAGE_LOCATIONS, key=f"{digest}_loc")
    added_date = st.date_input(
        "Added to fridge",
        value=date.today(),
        format="YYYY-MM-DD",
        key=f"{digest}_added_date",
    )
    st.caption(SAFETY)

    if st.session_state.get(f"{digest}_added"):
        st.success("Added to your fridge.")
        st.button("See my fridge", type="primary", on_click=goto, args=("Fridge",), key=f"{digest}_see")
        return
    chosen = [p for p in picks if p[0]]
    if st.button(f"Add {len(chosen)} to my fridge", type="primary", disabled=not chosen, key=f"{digest}_go"):
        try:
            for _, name, category, qty, pred, detection, detected_count in chosen:
                label = pred.label if pred and not pred.uncertain else None
                item_id = inv.add_item(
                    name=name, category=category, quantity=qty, storage_location=where,
                    date_added=added_date.isoformat(),
                    freshness_label=label, freshness_confidence=pred.confidence if label else None,
                    image_data=ip.thumbnail_jpeg(image),
                )
                inv.db.add_scan(
                    "detection",
                    f"{name.strip()} added (detected {detected_count})",
                    {
                        "freshness": label,
                        "suggested_food": detection.label,
                        "confirmed_food": name.strip(),
                        "confidence": detection.confidence,
                        "detected_count": detected_count,
                        "detector_source": detection.source,
                        "dataset_quality_label": detection.quality,
                    },
                    item_id,
                    image_data=ip.thumbnail_jpeg(image),
                )
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state[f"{digest}_added"] = True
        st.rerun()


def _manual_form(inv, classifier, classifier_ready: bool, image, digest: str, expanded: bool = False) -> None:
    with st.expander("Add it by hand", expanded=expanded):
        with st.form(f"manual_{digest}", clear_on_submit=True):
            c1, c2 = st.columns(2)
            name = c1.text_input("Name", placeholder="Whole milk")
            category = c2.selectbox("Category", settings.FOOD_CATEGORIES)
            c3, c4, c5 = st.columns(3)
            qty = c3.number_input("How many", min_value=0.0, value=1.0, step=1.0)
            unit = c4.selectbox("Unit", settings.UNITS)
            where = c5.selectbox("Where will it go?", settings.STORAGE_LOCATIONS)
            c6, c7 = st.columns(2)
            kind = c6.selectbox("Package date type", ["No date"] + list(MANUAL_PACKAGE_DATE_TYPES),
                                format_func=lambda k: k if k == "No date" else settings.DATE_TYPE_LABELS[k])
            when = c7.date_input("Enter selected date", value=date.today() + timedelta(days=7), format="YYYY-MM-DD")
            added_date = st.date_input("Added to fridge", value=date.today(), format="YYYY-MM-DD")
            use_photo = False
            if image is not None and classifier_ready and category == "Fruits & Vegetables":
                use_photo = st.checkbox("Check visible freshness from this photo (only trained food types)")
            if st.form_submit_button("Add to fridge", type="primary"):
                label = conf = None
                if use_photo:
                    try:
                        if classifier.supports_food(name):
                            pred = classifier.predict(ip.bgr_to_rgb(image))
                            if not pred.uncertain:
                                label, conf = pred.label, pred.confidence
                            else:
                                st.info(f"The photo was too unclear to judge freshness (best guess: {pred.display.lower()}).")
                        else:
                            supported = ", ".join(classifier.supported_foods())
                            st.info(
                                f"The trained freshness model does not support {name.strip() or 'this food'} yet. "
                                f"Currently supported: {supported}."
                            )
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Freshness check failed: {exc}")
                try:
                    dates = [] if kind == "No date" else [dict(date_type=kind, date_value=when.isoformat(), verified=True)]
                    item_id = inv.add_item(name=name, category=category, quantity=qty, unit=unit,
                                           storage_location=where, date_added=added_date.isoformat(),
                                           freshness_label=label, freshness_confidence=conf,
                                           image_data=ip.thumbnail_jpeg(image) if image is not None else None,
                                           dates=dates)
                    inv.db.add_scan(
                        "detection",
                        f"{name.strip()} added by hand",
                        {"freshness": label},
                        item_id,
                        image_data=ip.thumbnail_jpeg(image) if image is not None else None,
                    )
                    st.success(f"Added {name.strip()} to your fridge.")
                except ValueError as exc:
                    st.error(str(exc))


# ----------------------------------------------------------------------------
# Package date
# ----------------------------------------------------------------------------
def _target_picker(inv, digest: str) -> dict:
    items = inv.db.list_items()
    options = ["A new item"] + [f"{i['name']} (added {i['date_added']})" for i in items]
    choice = st.selectbox("Which item is this date for?", range(len(options)), format_func=lambda i: options[i], key="date_target")
    if choice == 0:
        c1, c2 = st.columns(2)
        name = c1.text_input("Item name", key="date_new_name", placeholder="Greek yogurt")
        category = c2.selectbox("Category", settings.FOOD_CATEGORIES, key="date_new_cat")
        return dict(item=None, name=name, category=category)
    return dict(item=items[choice - 1], name=items[choice - 1]["name"], category=items[choice - 1]["category"])


def _resolve_item(inv, target: dict, digest: str) -> int:
    """Existing item id, or create the new item once and reuse it for later dates in this scan."""
    if target["item"]:
        return target["item"]["id"]
    if not target["name"].strip():
        raise ValueError("Type the item's name first.")
    created = st.session_state.setdefault("created_for", {})
    key = f"{digest}|{target['name'].strip().lower()}"
    if key not in created:
        created[key] = inv.add_item(name=target["name"], category=target["category"])
    return created[key]


def _say(d: date) -> str:
    return f"{d.day} {d:%B %Y}"


def _candidate_card(
    inv,
    digest: str,
    index: int,
    cand: DateCandidate,
    target: dict,
    image_data: Optional[bytes] = None,
) -> None:
    done = st.session_state.get(f"{digest}_{index}_done")
    with st.container(border=True):
        if done:
            st.success("Saved." if done == "saved" else "Skipped.")
            return
        left, right = st.columns([1, 1.25], gap="large")
        with left:
            label = cand.label_text if cand.label_text else "No label found next to it"
            conf = f"Scanner confidence {cand.ocr_confidence:.0f}%" if cand.ocr_confidence is not None else "No confidence value"
            st.markdown(f'<div class="found"><div class="big">{theme.esc(label)}</div>'
                        f'<div class="small">Printed as <span class="readout">{theme.esc(cand.raw_text)}</span></div>'
                        f'<div class="small">{theme.esc(conf)}</div></div>', unsafe_allow_html=True)
            if cand.note:
                st.caption(cand.note)
            if cand.ocr_confidence is not None and cand.ocr_confidence < 60:
                st.caption("Low confidence. Check this one against the package.")
        with right:
            types = ["Choose a type"] + list(settings.DATE_TYPES)
            start = types.index(cand.label_type) if cand.label_type in types else 0
            kind = st.selectbox("What kind of date is it?", types, index=start, key=f"{digest}_{index}_type",
                                format_func=lambda k: k if k == "Choose a type" else settings.DATE_TYPE_LABELS[k])
            if cand.ambiguous:
                st.caption("This could be read two ways. Pick the right one.")
                notes = ["day first", "month first"]
                chosen = st.radio("Date", cand.candidates, index=None,
                                  format_func=lambda d: f"{_say(d)} ({notes[cand.candidates.index(d)]})",
                                  key=f"{digest}_{index}_pick", label_visibility="collapsed")
            else:
                chosen = st.date_input("Enter selected date", value=cand.parsed, format="YYYY-MM-DD", key=f"{digest}_{index}_date")
            b1, b2 = st.columns(2)
            save = b1.button("Confirm and save", type="primary", key=f"{digest}_{index}_save")
            skip = b2.button("Not right", key=f"{digest}_{index}_skip")
        if save:
            if kind == "Choose a type":
                st.warning("Choose what kind of date it is first.")
            elif chosen is None:
                st.warning("Pick the date first.")
            else:
                try:
                    item_id = _resolve_item(inv, target, digest)
                    inv.add_date(item_id, kind, chosen.isoformat(), source="ocr", verified=True,
                                 original_label=cand.label_text, original_text=cand.raw_text, ocr_confidence=cand.ocr_confidence)
                    inv.db.add_scan(
                        "ocr",
                        f"{settings.DATE_TYPE_LABELS[kind]} saved for {target['name']}",
                        {"raw": cand.raw_text},
                        item_id,
                        image_data=image_data,
                    )
                    st.session_state[f"{digest}_{index}_done"] = "saved"
                    st.rerun()
                except ValueError as exc:
                    st.warning(str(exc))
        if skip:
            inv.db.add_scan("ocr", f"Rejected a read of {cand.raw_text}", {"raw": cand.raw_text}, status="rejected")
            st.session_state[f"{digest}_{index}_done"] = "skipped"
            st.rerun()


def _dates() -> None:
    inv, engine = get_inventory(), get_ocr_engine()
    status = engine.status()
    scan_mode = st.radio(
        "Package date scan mode",
        ["Live camera", "Photo or upload"],
        horizontal=True,
        key="date_scan_mode",
    )
    data = _image_input("date") if scan_mode == "Photo or upload" else None
    if not status.available:
        st.error(f"{status.message}. You can still type a date in by hand below.")
    theme.note("A date only counts toward alerts after you confirm it. Manufacturing and packing dates never count as an expiry.", "info")
    digest = hashlib.md5(data).hexdigest() if data else ("live-date" if scan_mode == "Live camera" else "manual")
    target = _target_picker(inv, digest)

    if scan_mode == "Live camera" and status.available:
        detector, classifier = get_detector(), get_classifier()
        freshness_status = classifier.status()
        camera_column, result_column = st.columns([1.2, 1], gap="medium")
        with camera_column:
            st.caption("Keep the package label in view. The camera reads dates continuously.")
            with st.container(key="live-date-camera"):
                ctx = webrtc_streamer(
                    key="freshtrack-live-date",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration={"iceServers": []},
                    frontend_rtc_configuration={"iceServers": []},
                    video_processor_factory=lambda: LiveFoodProcessor(
                        detector,
                        classifier,
                        engine,
                        freshness_status.available,
                        True,
                    ),
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 1280},
                            "height": {"ideal": 720},
                            "frameRate": {"ideal": 15, "max": 30},
                        },
                        "audio": False,
                    },
                    async_processing=True,
                    video_html_attrs={
                        "autoPlay": True,
                        "controls": True,
                        "muted": True,
                        "style": {"width": "100%", "height": "auto", "objectFit": "contain"},
                    },
                )
        with result_column:
            _live_date_review(ctx, inv, target)

    image = None
    if data and status.available:
        try:
            image = ip.load_image(data)
        except Exception:  # noqa: BLE001
            st.error("That file could not be read as an image. Try a JPG or PNG.")
            return
        cache = st.session_state.setdefault("ocr_cache", {})
        if digest not in cache:
            with st.spinner("Reading the label..."):
                cache[digest] = read_expiration_dates(image, engine)
        result = cache[digest]

        theme.section("What the scanner found")
        if result.status != "ok":
            theme.note(result.message, "info")
        else:
            for i, cand in enumerate(result.candidates):
                _candidate_card(inv, digest, i, cand, target, ip.thumbnail_jpeg(image))
        if result.raw_text:
            with st.expander("Everything the scanner read"):
                st.code(result.raw_text, language=None)
        st.caption("Scanners miss dates on curved, shiny, faded or blurry packaging. Check each one against the package.")

    with st.expander("Enter a date by hand", expanded=not data):
        c1, c2 = st.columns(2)
        kind = c1.selectbox(
            "Package date type",
            MANUAL_PACKAGE_DATE_TYPES,
            format_func=lambda k: settings.DATE_TYPE_LABELS[k],
            key="hand_kind",
        )
        when = c2.date_input("Enter selected date", value=date.today() + timedelta(days=7), format="YYYY-MM-DD", key="hand_date")
        if st.button("Save date", type="primary", key="hand_save"):
            try:
                item_id = _resolve_item(inv, target, digest)
                inv.add_date(item_id, kind, when.isoformat(), source="manual", verified=True)
                inv.db.add_scan(
                    "ocr",
                    f"{settings.DATE_TYPE_LABELS[kind]} typed in for {target['name']}",
                    item_id=item_id,
                    image_data=ip.thumbnail_jpeg(image) if image is not None else None,
                )
                st.success("Saved.")
            except ValueError as exc:
                st.warning(str(exc))


@st.fragment(run_every="1s")
def _live_date_review(ctx, inv, target: dict) -> None:
    processor = ctx.video_processor
    if processor is None:
        theme.section("Live food recognition")
        st.caption("Start the camera to see detected foods alongside package-date readings.")
        return

    for error in processor.latest_errors():
        st.warning(error)

    _, food_observations = processor.latest_observation()
    theme.section("Live food recognition")
    if food_observations:
        for detection, prediction in food_observations:
            st.success(
                f"{detection.label.title()} · food-detection confidence {detection.confidence:.0%}"
            )
            freshness_info = _live_freshness_info(detection, prediction)
            if freshness_info:
                st.caption(freshness_info)
    else:
        st.caption("No supported food detected in the current view.")

    raw_text, candidates = processor.latest_date_observation()
    pending = st.session_state.get("live_date_candidates", [])
    digest = st.session_state.get("live_date_digest", "")
    has_unresolved = bool(
        pending
        and any(not st.session_state.get(f"{digest}_{index}_done") for index in range(len(pending)))
    )
    if candidates and not has_unresolved and not st.session_state.get("live_date_latched", False):
        signature = "|".join(
            f"{candidate.label_type}:{candidate.raw_text}:{candidate.line_text}"
            for candidate in candidates
        )
        st.session_state["live_date_signature"] = signature
        st.session_state["live_date_candidates"] = candidates
        st.session_state["live_date_digest"] = hashlib.md5(signature.encode()).hexdigest()
        st.session_state["live_date_latched"] = True

    pending = st.session_state.get("live_date_candidates", [])
    if pending:
        digest = st.session_state["live_date_digest"]
        pending_done = all(
            st.session_state.get(f"{digest}_{index}_done")
            for index in range(len(pending))
        )
        if pending_done and not candidates:
            st.session_state["live_date_latched"] = False
        theme.section("Live date readings", "Confirm each printed date and its label before saving.")
        for index, candidate in enumerate(pending):
            _candidate_card(
                inv,
                digest,
                index,
                candidate,
                target,
                processor.latest_date_thumbnail(),
            )
    elif raw_text:
        st.caption(f"Reading package text: {raw_text}")
        st.info("No date has been confidently read from the current view yet. Hold the printed date closer and steady.")
    else:
        st.caption("Waiting for a readable package label. Hold the printed date close, steady and well lit.")
