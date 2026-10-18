"""Deliberately synthetic retail fixture for APP_MODE=demo and tests.

These products and counts are invented. They are never mixed with the real CSV model.
"""

from __future__ import annotations

from .model import RetailModel
from .service import RetailModelService

DEMO_CATALOG = {
    "FX-CAMERA": "Fixture IP Camera",
    "FX-DETECTOR": "Fixture Motion Detector",
    "FX-NVR": "Fixture Network Video Recorder",
    "FX-PANEL": "Fixture Control Panel",
    "FX-RECEIVER": "Fixture Wireless Receiver",
}

# 100 synthetic baskets: FX-PANEL in 40, FX-RECEIVER in 10 (8 with FX-PANEL).
DEMO_BASKETS = (
    [["FX-PANEL", "FX-RECEIVER"]] * 8
    + [["FX-RECEIVER", "FX-CAMERA"]] * 2
    + [["FX-PANEL", "FX-DETECTOR"]] * 32
    + [["FX-CAMERA", "FX-NVR"]] * 58
)

DEMO_WARNING = "Synthetic demonstration counts; not measured from the supplied CSV."


def build_demo_model() -> RetailModel:
    transactions = {str(i): basket for i, basket in enumerate(DEMO_BASKETS, start=1)}
    return RetailModel.train(transactions, DEMO_CATALOG, list(transactions), model_version="demo-retail-v1")


def build_demo_service() -> RetailModelService:
    return RetailModelService(build_demo_model(), mode="demo", warnings=[DEMO_WARNING])
