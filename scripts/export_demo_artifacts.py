# -*- coding: utf-8 -*-
"""Build the two artifacts consumed by the portfolio's in-browser demo.

The demo at https://juleescourne.github.io/portfolio-data-analyst/#/churn runs the
trained classifier client-side. It needs two files that are *not* versioned here,
because they are build outputs rather than source:

1. ``xgb_churn_model.onnx``  — the model, so onnxruntime-web can score in the browser.
2. ``shap_lookup.json``      — SHAP values pre-computed over the demo's input grid.

SHAP cannot run in the browser, so the explanation panel reads from this lookup. The
grid is the exact cartesian product the demo exposes through its controls, which is
why the front-end can round any user input to the nearest grid point.

Run it after ``04_modeling.ipynb`` has saved the model:

    pip install -r requirements.txt shap onnxmltools onnxconverter-common
    python scripts/export_demo_artifacts.py

Outputs land in ``model/`` by default.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# The nine features kept by the reduced model, in the order the classifier expects.
FEATURES = [
    "age",
    "num_of_products",
    "is_active_member",
    "geography_Germany",
    "product_1_inactive",
    "product_engagement_score",
    "medium_balance_risk",
    "product_1_AND_germany",
    "product_1_AND_female",
]

# Grid mirroring the demo's controls. Changing a step here changes the rounding the
# front-end must apply in useShapValues.jsx — keep the two in sync.
AGES = list(range(18, 103, 5))            # 18, 23, … 98   (step 5)
BALANCES = list(range(0, 260_000, 10_000))  # 0 … 250 000   (step 10 000)
PRODUCTS = [1, 2, 3, 4]
GEOGRAPHIES = ["France", "Allemagne", "Espagne"]
GENDERS = ["Homme", "Femme"]
ACTIVE = [0, 1]

TOP_N = 6  # the demo displays the five strongest, one spare for ties


def build_grid() -> pd.DataFrame:
    """Return one row per demo scenario, with both the labels and the model features."""
    rows = []
    for products, geography, age, gender, balance, is_active in product(
        PRODUCTS, GEOGRAPHIES, AGES, GENDERS, BALANCES, ACTIVE
    ):
        is_germany = int(geography == "Allemagne")
        is_female = int(gender == "Femme")
        is_single_product = int(products == 1)

        rows.append(
            {
                "key": f"{products}|{geography}|{age}|{gender}|{balance}|{is_active}",
                "age": age,
                "num_of_products": products,
                "is_active_member": is_active,
                "geography_Germany": is_germany,
                "product_1_inactive": int(is_single_product and is_active == 0),
                "product_engagement_score": products * is_active,
                "medium_balance_risk": int(100_000 <= balance <= 140_000),
                "product_1_AND_germany": int(is_single_product and is_germany),
                "product_1_AND_female": int(is_single_product and is_female),
            }
        )
    return pd.DataFrame(rows)


def export_shap_lookup(model, grid: pd.DataFrame, destination: Path) -> int:
    """Compute SHAP values for every grid row and write the lookup table."""
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(grid[FEATURES])

    lookup: dict[str, list[dict]] = {}
    for row_index, key in enumerate(grid["key"]):
        row = values[row_index]
        order = np.argsort(np.abs(row))[::-1][:TOP_N]
        lookup[key] = [
            {
                "feature": FEATURES[i],
                "shap_value": round(float(row[i]), 4),
                "abs_value": round(float(abs(row[i])), 4),
            }
            for i in order
        ]

    destination.write_text(json.dumps(lookup, separators=(",", ":")), encoding="utf-8")
    return len(lookup)


def export_onnx(model, destination: Path) -> None:
    """Convert the XGBoost classifier to ONNX for onnxruntime-web."""
    from onnxconverter_common.data_types import FloatTensorType
    from onnxmltools.convert import convert_xgboost

    onnx_model = convert_xgboost(
        model,
        initial_types=[("input", FloatTensorType([None, len(FEATURES)]))],
        target_opset=12,
    )
    destination.write_bytes(onnx_model.SerializeToString())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("model/xgb_churn_model.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("model"))
    parser.add_argument("--skip-onnx", action="store_true", help="only rebuild the SHAP lookup")
    args = parser.parse_args()

    if not args.model.exists():
        parser.error(
            f"{args.model} not found. Run notebooks/04_modeling.ipynb first — it saves the model."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(args.model)

    grid = build_grid()
    print(f"Grid: {len(grid):,} scenarios over {len(FEATURES)} features")

    lookup_path = args.output_dir / "shap_lookup.json"
    count = export_shap_lookup(model, grid, lookup_path)
    print(f"SHAP lookup written to {lookup_path} ({count:,} entries)")

    if not args.skip_onnx:
        onnx_path = args.output_dir / "xgb_churn_model.onnx"
        export_onnx(model, onnx_path)
        print(f"ONNX model written to {onnx_path}")

    print("\nUpload both files to the portfolio's `assets` branch, under models/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
