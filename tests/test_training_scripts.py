"""Tests for the dataset tooling (no model download, tiny synthetic data)."""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

import prepare_dataset as prep  # noqa: E402
import prepare_freshness44 as prep_freshness44  # noqa: E402
import prepare_fresh_stale as prep_fresh_stale  # noqa: E402
import prepare_lvis_produce as prep_lvis  # noqa: E402
import evaluate_lvis_produce as eval_lvis  # noqa: E402


def noise_image(seed, size=64):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def test_hash_is_flip_aware_and_identical_images_match():
    img = noise_image(1)
    h = prep.dhash(img)
    assert prep.hamming(h, np.array([prep.dhash(img)]))[0] == 0
    # A mirrored copy should match the mirrored hash of the original.
    mirrored = img.transpose(Image.FLIP_LEFT_RIGHT)
    assert prep.hamming(prep.dhash(mirrored), np.array([prep.dhash(img.transpose(Image.FLIP_LEFT_RIGHT))]))[0] == 0
    assert prep.hamming(prep.dhash(noise_image(2)), np.array([h]))[0] > 40


def test_groups_join_same_photo_and_near_duplicates():
    objs = [dict(image_id="a"), dict(image_id="a"), dict(image_id="b"), dict(image_id="c")]
    base, other = noise_image(1), noise_image(7)
    hashes = np.array([prep.dhash(base), prep.dhash(other), prep.dhash(base), prep.dhash(noise_image(9))])
    flipped = np.array([prep.dhash(i.transpose(Image.FLIP_LEFT_RIGHT)) for i in (base, other, base, noise_image(9))])
    g = prep.group_ids(objs, hashes, flipped, threshold=10)
    assert g[0] == g[1]          # same photo "a"
    assert g[0] == g[2]          # b duplicates a's first crop
    assert g[3] != g[0]          # c is unrelated


def test_split_never_shares_a_group():
    rng = np.random.default_rng(0)
    objs = [dict(image_id=str(i // 2), label=("fresh", "aging", "spoiled")[i % 3], fruit="apple") for i in range(300)]
    groups = np.array([i // 2 for i in range(300)])
    prep.split_by_group(objs, groups, 0.15, 0.15, seed=1)
    seen = {}
    for o, g in zip(objs, groups):
        assert seen.setdefault(g, o["split"]) == o["split"]
    assert {o["split"] for o in objs} == {"train", "val", "test"}


def make_dataset(root: Path, per_class=60, classes=("fresh", "aging", "spoiled")):
    n = 0
    for split, count in (("train", per_class), ("val", 12), ("test", 12)):
        for cls in classes:
            folder = root / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            for _ in range(count):
                noise_image(n).save(folder / f"{n}.jpg")
                n += 1


def run_check(path):
    return subprocess.run([sys.executable, str(ROOT / "training" / "check_freshness_dataset.py"), "--data", str(path)],
                          capture_output=True, text=True)


def test_dataset_checker_accepts_a_good_dataset(tmp_path):
    make_dataset(tmp_path)
    r = run_check(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Dataset check passed" in r.stdout


def test_dataset_checker_flags_missing_class_and_leakage(tmp_path):
    make_dataset(tmp_path)
    import shutil
    shutil.rmtree(tmp_path / "val" / "spoiled")
    shutil.copy(next((tmp_path / "train" / "fresh").iterdir()), tmp_path / "test" / "fresh" / "leak.jpg")
    r = run_check(tmp_path)
    assert r.returncode == 1
    assert "Missing folder" in r.stdout and "Identical file" in r.stdout


def _make_base_freshness_dataset(root: Path) -> None:
    rows = []
    image_id = 0
    for split in ("train", "val", "test"):
        for label in ("fresh", "aging", "spoiled"):
            relative = Path(split) / label / f"{image_id}.jpg"
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            noise_image(image_id).save(target)
            rows.append({
                "path": relative.as_posix(),
                "split": split,
                "label": label,
                "fruit": "apple",
                "image_id": str(image_id),
            })
            image_id += 1
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "split", "label", "fruit", "image_id"])
        writer.writeheader()
        writer.writerows(rows)


def test_freshness44_merge_keeps_three_classes_and_records_food_metadata(tmp_path):
    base = tmp_path / "base"
    source = tmp_path / "freshness44"
    output = tmp_path / "expanded"
    _make_base_freshness_dataset(base)
    for folder in ("Apple_Fresh", "Tomato_Rotten"):
        (source / folder).mkdir(parents=True)
        for index in range(10):
            noise_image(len(folder) + index).save(source / folder / f"{index}.jpg")

    counts = prep_freshness44.prepare_freshness44(source, base, output)

    assert counts["train"] > 0 and counts["val"] > 0 and counts["test"] > 0
    assert all((output / split / label).is_dir()
               for split in ("train", "val", "test")
               for label in ("fresh", "aging", "spoiled"))
    with (output / "manifest.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {"apple", "tomato"} <= {row["fruit"] for row in rows}
    assert "CC BY-SA 4.0" in (output / "ATTRIBUTION.txt").read_text(encoding="utf-8")


def test_freshness44_folder_labels_are_normalized():
    assert prep_freshness44.parse_quality_folder("Apple_Fresh") == ("apple", "fresh")
    assert prep_freshness44.parse_quality_folder("Bitter_Gourd_Rotten") == ("bitter gourd", "spoiled")
    assert prep_freshness44.parse_quality_folder("Mixed_Quality") is None


def test_fresh_stale_paths_find_food_and_condition():
    source = Path("/datasets/fresh-stale")
    assert prep_fresh_stale.parse_image_path(
        source / "Fresh" / "Bitter_Gourd" / "image.jpg", source
    ) == ("bitter gourd", "fresh")
    assert prep_fresh_stale.parse_image_path(
        source / "staleApples" / "image.jpg", source
    ) == ("apple", "stale")
    assert prep_fresh_stale.parse_image_path(
        source / "fresh" / "unknown_food" / "image.jpg", source
    ) is None


def test_fresh_stale_merge_maps_stale_to_aging_and_keeps_duplicate_groups_together(tmp_path):
    base = tmp_path / "base"
    source = tmp_path / "fresh-stale"
    output = tmp_path / "expanded"
    _make_base_freshness_dataset(base)
    seed = 1000
    for food in prep_fresh_stale.FOODS:
        folder_name = food.replace(" ", "_")
        for condition in ("fresh", "stale"):
            folder = source / f"{condition}_{folder_name}"
            folder.mkdir(parents=True)
            for index in range(30):
                noise_image(seed).save(folder / f"{index}.jpg")
                seed += 1

    similar_folder = source / "fresh_apple"
    original = noise_image(9000)
    original.save(similar_folder / "video-frame-a.jpg", quality=95)
    original.save(similar_folder / "video-frame-b.jpg", quality=80)

    counts = prep_fresh_stale.prepare_fresh_stale(source, base, output)

    assert counts["train"] > 0 and counts["val"] > 0 and counts["test"] > 0
    with (output / "manifest.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    added = [row for row in rows if row["source"] == prep_fresh_stale.SOURCE_NAME]
    assert {row["fruit"] for row in added} == set(prep_fresh_stale.FOODS)
    assert "aging" in {row["label"] for row in added}
    video_frames = [row for row in added if row["source_file"].startswith("fresh_apple/video-frame-")]
    assert len(video_frames) == 2
    assert len({row["split"] for row in video_frames}) == 1
    attribution = (output / "ATTRIBUTION.txt").read_text(encoding="utf-8")
    assert "CC0" in attribution and "third-party content" in attribution


def test_lvis_preparation_corrects_validation_path_and_keeps_test_separate(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "prepared" / "data.yaml"
    source.mkdir()
    classes = [f"food {index}" for index in range(prep_lvis.EXPECTED_CLASSES)]
    classes[6] = "banana"
    classes[24] = "cucumber/cuke"
    (source / "data.yaml").write_text(yaml.safe_dump({"names": classes}), encoding="utf-8")

    for split, nested in (("train", "train"), ("val", "val"), ("test", "")):
        image_dir = source / "images" / split / nested
        label_dir = source / "labels" / split / nested
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / "sample.jpg").write_bytes(b"image")
        (label_dir / "sample.txt").write_text("24 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    counts = prep_lvis.prepare_dataset(source, output)
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert counts == {"train": 1, "val": 1, "test": 1}
    assert config["path"] == str(source.resolve())
    assert config["val"] == "images/val/val"
    assert config["test"] == "images/test"
    assert "LVIS" in (output.parent / "ATTRIBUTION.txt").read_text(encoding="utf-8")


def test_lvis_preparation_rejects_bad_annotation_class_id(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    names = [f"food {index}" for index in range(prep_lvis.EXPECTED_CLASSES)]
    names[6], names[24] = "banana", "cucumber"
    (source / "data.yaml").write_text(yaml.safe_dump({"names": names}), encoding="utf-8")
    for split, nested in (("train", "train"), ("val", "val"), ("test", "")):
        image_dir = source / "images" / split / nested
        label_dir = source / "labels" / split / nested
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / "sample.jpg").write_bytes(b"image")
        class_id = prep_lvis.EXPECTED_CLASSES if split == "train" else 24
        (label_dir / "sample.txt").write_text(f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Out-of-range"):
        prep_lvis.prepare_dataset(source, tmp_path / "prepared" / "data.yaml")


def test_lvis_class_ap_uses_compact_metric_class_axis():
    class BoxMetrics:
        ap_class_index = np.array([1, 6, 24])
        ap50 = np.array([0.45, 0.83, 0.27])
        maps = np.zeros(63)
        maps[1], maps[6], maps[24] = 0.33, 0.50, 0.13

    assert eval_lvis.class_average_precision(BoxMetrics, 6) == (0.83, 0.5)
    assert eval_lvis.class_average_precision(BoxMetrics, 24) == (0.27, 0.13)
    assert eval_lvis.class_average_precision(BoxMetrics, 2) is None
