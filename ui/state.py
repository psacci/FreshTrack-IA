"""Shared, cached resources and small helpers used by all pages."""
from __future__ import annotations

import streamlit as st

from config import settings
from modules.expiration_date_reader import TesseractEngine
from modules.food_detector import FoodDetector
from modules.freshness_classifier import FreshnessClassifier
from modules.inventory_manager import InventoryManager
from modules.roboflow_workflow import RoboflowWorkflow


@st.cache_resource
def get_inventory() -> InventoryManager:
    return InventoryManager()


@st.cache_resource
def get_detector() -> FoodDetector:
    return FoodDetector()


@st.cache_resource
def get_classifier() -> FreshnessClassifier:
    return FreshnessClassifier()


@st.cache_resource
def get_ocr_engine() -> TesseractEngine:
    return TesseractEngine()


@st.cache_resource
def get_roboflow_detector() -> RoboflowWorkflow | None:
    if not settings.ROBOFLOW_API_KEY.strip():
        return None
    return RoboflowWorkflow(settings.ROBOFLOW_API_KEY)


def goto(page: str, **extra) -> None:
    """Button callback that switches the top navigation (and optionally sets other state)."""
    st.session_state["page"] = page
    st.session_state.update(extra)
