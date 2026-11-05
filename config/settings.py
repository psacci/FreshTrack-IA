"""Central configuration for FreshTrack AI.

Everything that might change between machines or experiments lives here so
the rest of the code never hardcodes paths or thresholds. Values can be
overridden with environment variables or a `.env` file (see `.env.example`).
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines) so we avoid an extra dependency."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
DB_PATH = _path_from_env("FRESHTRACK_DB", DATA_DIR / "freshtrack.db")
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"
STYLES_PATH = BASE_DIR / "assets" / "styles.css"

MODELS_DIR = BASE_DIR / "models"


def _default_yolo_weights() -> Path:
    produce_weights = MODELS_DIR / "food_detection" / "freshtrack_lvis.pt"
    if produce_weights.is_file():
        return produce_weights
    return MODELS_DIR / "food_detection" / "yolov8n.pt"


YOLO_WEIGHTS = _path_from_env("FRESHTRACK_YOLO", _default_yolo_weights())
FRESHNESS_CHECKPOINT = _path_from_env(
    "FRESHTRACK_FRESHNESS_CKPT",
    MODELS_DIR / "freshness_classification" / "mobilenetv2_freshness.pt",
)
FRESHNESS_METRICS = MODELS_DIR / "freshness_classification" / "metrics.json"
WORLD_FOOD_WEIGHTS = _path_from_env(
    "FRESHTRACK_WORLD_FOOD",
    MODELS_DIR / "food_detection" / "yolov8s-worldv2.pt",
)
WORLD_FOOD_PROMPTS = (
    "beef steak", "ground beef", "chicken", "pork",
    "bacon", "sausage", "ham", "turkey", "hot dog",
    "fish", "tuna", "shrimp", "crab", "lobster",
    "egg", "milk", "cheese", "yogurt", "butter", "cream", "ice cream",
    "bread", "pasta", "noodles", "rice", "cereal", "beans", "tofu",
    "pizza", "sandwich", "cake", "donut",
)

DATASET_DIR = BASE_DIR / "dataset" / "freshness"

# --------------------------------------------------------------------------
# Food categories (all six are supported by inventory + date monitoring)
# --------------------------------------------------------------------------
FOOD_CATEGORIES = (
    "Fruits & Vegetables",
    "Dairy",
    "Meat & Seafood",
    "Bakery",
    "Canned & Preserved",
    "Dry & Packaged",
)
STORAGE_LOCATIONS = ("Fridge", "Freezer", "Pantry", "Counter")
UNITS = ("pcs", "pack", "g", "kg", "ml", "L", "can", "bottle")

# --------------------------------------------------------------------------
# Detection labels accepted from COCO, VegQual, and custom produce models.
# --------------------------------------------------------------------------
DETECTION_MIN_CONFIDENCE = 0.25
YOLO_TO_CATEGORY = {
    "apple": "Fruits & Vegetables",
    "almond": "Dry & Packaged",
    "apricot": "Fruits & Vegetables",
    "artichoke": "Fruits & Vegetables",
    "asparagus": "Fruits & Vegetables",
    "banana": "Fruits & Vegetables",
    "blackberry": "Fruits & Vegetables",
    "blueberry": "Fruits & Vegetables",
    "brussels sprouts": "Fruits & Vegetables",
    "cantaloupe": "Fruits & Vegetables",
    "capsicum": "Fruits & Vegetables",
    "carrot": "Fruits & Vegetables",
    "cauliflower": "Fruits & Vegetables",
    "cayenne pepper": "Fruits & Vegetables",
    "celery": "Fruits & Vegetables",
    "cherry": "Fruits & Vegetables",
    "chickpea": "Dry & Packaged",
    "chili pepper": "Fruits & Vegetables",
    "clementine": "Fruits & Vegetables",
    "coconut": "Fruits & Vegetables",
    "corn": "Fruits & Vegetables",
    "cucumber": "Fruits & Vegetables",
    "date": "Fruits & Vegetables",
    "eggplant": "Fruits & Vegetables",
    "fig": "Fruits & Vegetables",
    "garlic": "Fruits & Vegetables",
    "ginger": "Fruits & Vegetables",
    "gourd": "Fruits & Vegetables",
    "green bean": "Fruits & Vegetables",
    "green onion": "Fruits & Vegetables",
    "kiwi": "Fruits & Vegetables",
    "mandarin orange": "Fruits & Vegetables",
    "melon": "Fruits & Vegetables",
    "mushroom": "Fruits & Vegetables",
    "orange": "Fruits & Vegetables",
    "papaya": "Fruits & Vegetables",
    "pea": "Fruits & Vegetables",
    "peach": "Fruits & Vegetables",
    "pear": "Fruits & Vegetables",
    "persimmon": "Fruits & Vegetables",
    "pickle": "Canned & Preserved",
    "pineapple": "Fruits & Vegetables",
    "prune": "Fruits & Vegetables",
    "pumpkin": "Fruits & Vegetables",
    "radish": "Fruits & Vegetables",
    "raspberry": "Fruits & Vegetables",
    "strawberry": "Fruits & Vegetables",
    "sweet potato": "Fruits & Vegetables",
    "tofu": "Dry & Packaged",
    "turnip": "Fruits & Vegetables",
    "watermelon": "Fruits & Vegetables",
    "broccoli": "Fruits & Vegetables",
    "tomato": "Fruits & Vegetables",
    "potato": "Fruits & Vegetables",
    "bitter gourd": "Fruits & Vegetables",
    "pointed gourd": "Fruits & Vegetables",
    "onion": "Fruits & Vegetables",
    "brinjal": "Fruits & Vegetables",
    "lettuce": "Fruits & Vegetables",
    "spinach": "Fruits & Vegetables",
    "cabbage": "Fruits & Vegetables",
    "zucchini": "Fruits & Vegetables",
    "mango": "Fruits & Vegetables",
    "grape": "Fruits & Vegetables",
    "avocado": "Fruits & Vegetables",
    "lemon": "Fruits & Vegetables",
    "lime": "Fruits & Vegetables",
    "egg": "Dairy",
    "eggs": "Dairy",
    "milk": "Dairy",
    "yogurt": "Dairy",
    "cheese": "Dairy",
    "butter": "Dairy",
    "cake": "Bakery",
    "donut": "Bakery",
    "bread": "Bakery",
    "sandwich": "Bakery",
    "hot dog": "Meat & Seafood",
    "pizza": "Bakery",
    "chicken": "Meat & Seafood",
    "beef": "Meat & Seafood",
    "pork": "Meat & Seafood",
    "fish": "Meat & Seafood",
    "salmon": "Meat & Seafood",
    "tuna": "Meat & Seafood",
    "rice": "Dry & Packaged",
    "pasta": "Dry & Packaged",
    "noodles": "Dry & Packaged",
    "cereal": "Dry & Packaged",
    "beans": "Canned & Preserved",
    "beef": "Meat & Seafood",
    "steak": "Meat & Seafood",
    "beef steak": "Meat & Seafood",
    "ground beef": "Meat & Seafood",
    "chicken breast": "Meat & Seafood",
    "pork chop": "Meat & Seafood",
    "sausage": "Meat & Seafood",
    "ham": "Meat & Seafood",
    "fish": "Meat & Seafood",
    "shrimp": "Meat & Seafood",
    "milk carton": "Dairy",
    "bread": "Bakery",
    "pasta": "Dry & Packaged",
    "bacon": "Meat & Seafood",
    "turkey": "Meat & Seafood",
    "crab": "Meat & Seafood",
    "lobster": "Meat & Seafood",
    "cream": "Dairy",
    "ice cream": "Dairy",
}
YOLO_LABEL_ALIASES = {
    "bean curd/tofu": "tofu",
    "bell pepper/capsicum": "capsicum",
    "cantaloup/cantaloupe": "cantaloupe",
    "cayenne/cayenne spice/cayenne pepper/cayenne pepper spice/red pepper/red pepper": "cayenne pepper",
    "chili/chili vegetable/chili pepper/chili pepper vegetable/chilli/chilli vegetable/chilly/chilly": "chili pepper",
    "chickpea/garbanzo": "chickpea",
    "coconut/cocoanut": "coconut",
    "cucumber/cuke": "cucumber",
    "date/date fruit": "date",
    "eggplant/aubergine": "eggplant",
    "fig/fig fruit": "fig",
    "garlic/ail": "garlic",
    "edible corn/corn/maize": "corn",
    "green onion/spring onion/scallion": "green onion",
    "ginger/gingerroot": "ginger",
    "kiwi fruit": "kiwi",
    "orange/orange fruit": "orange",
    "pea/pea food": "pea",
    "radish/daikon": "radish",
    "zucchini/courgette": "zucchini",
    "hotdog": "hot dog",
}

# Optional Roboflow Workflows inference. API keys are read only from the
# environment or the local .env file and are never stored in this module.
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "blanks-workspace-suk8j")
ROBOFLOW_WORKFLOW_ID = os.environ.get("ROBOFLOW_WORKFLOW_ID", "foods-testing-mkupr")
ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
ROBOFLOW_IMAGE_INPUT = os.environ.get("ROBOFLOW_IMAGE_INPUT", "image")
ROBOFLOW_WORKFLOW_PARAMETERS = os.environ.get("ROBOFLOW_WORKFLOW_PARAMETERS", "")

# --------------------------------------------------------------------------
# Visual freshness (MobileNetV2)
# --------------------------------------------------------------------------
FRESHNESS_CLASSES = ("fresh", "aging", "spoiled")
FRESHNESS_LABELS = {
    "fresh": "Fresh",
    "aging": "Aging / Overripe",
    "spoiled": "Visible spoilage",
}
FRESHNESS_FRUITS = ("apple", "banana", "orange")
FRESHNESS_IMAGE_SIZE = 224
FRESHNESS_MIN_CONFIDENCE = 0.60  # below this the app says "not sure"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------
# Dates and alerts
# --------------------------------------------------------------------------
DATE_TYPES = ("expiration", "use_by", "best_before", "sell_by", "manufacturing", "packing")
DATE_TYPE_LABELS = {
    "expiration": "Expiration date",
    "use_by": "Use-by date",
    "best_before": "Best-before date",
    "sell_by": "Sell-by date",
    "manufacturing": "Manufacturing date",
    "packing": "Packing date",
}
# Only these can trigger "past date" or "expiring soon" alerts.
ALERT_DATE_TYPES = ("expiration", "use_by", "best_before", "sell_by")
DEFAULT_SOON_DAYS = 3


def get_device() -> str:
    """Pick the best compute device: Apple MPS, then CUDA, then CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
