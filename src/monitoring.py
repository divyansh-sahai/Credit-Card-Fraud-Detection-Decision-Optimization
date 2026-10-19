
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def monitor_batch(
    reference_data,
    incoming_data,
    model,
    threshold,
    output_dir,
    psi_warning_threshold=0.20,
):
    """
    Monitor a batch of transactions against reference data.

    reference_data: Historical feature data used as the baseline.
    incoming_data: New transaction features without the target column.
    model: Trained model with predict_proba().
    threshold: Fraud decision threshold selected during validation.
    output_dir: Folder in which monitoring reports will be saved.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = reference_data.copy()
    incoming = incoming_data.copy()

    # Validate that the incoming batch has the required features.
    if list(reference.columns) != list(incoming.columns):
        raise ValueError(
            "Reference and incoming data must have identical "
            "feature columns in the same order."
        )

    if incoming.empty:
        raise ValueError("Incoming transaction batch cannot be empty.")

    if reference.empty:
        raise ValueError("Reference data cannot be empty.")

    if not np.isfinite(reference.to_numpy(dtype=float)).all():
        raise ValueError("Reference data contains missing or infinite values.")

    if not np.isfinite(incoming.to_numpy(dtype=float)).all():
        raise ValueError("Incoming data contains missing or infinite values.")

    # Generate model scores and threshold-based decisions.
    fraud_scores = model.predict_proba(incoming)[:, 1]
    predictions = (fraud_scores >= threshold).astype(int)

    # Record each prediction without storing unnecessary personal data.
    timestamp = datetime.now(timezone.utc).isoformat()

    prediction_log = pd.DataFrame({
        "prediction_timestamp_utc": [timestamp] * len(incoming),
        "fraud_probability": fraud_scores,
        "fraud_prediction": predictions,
        "risk_classification": np.where(
            predictions == 1, "Fraud", "Legitimate"
        ),
    })

    # Summarize the incoming batch.
    batch_summary = pd.DataFrame([{
        "monitoring_timestamp_utc": timestamp,
        "transactions_monitored": len(incoming),
        "transactions_flagged": int(predictions.sum()),
        "flagged_transaction_rate": float(predictions.mean()),
        "mean_fraud_probability": float(fraud_scores.mean()),
        "median_fraud_probability": float(np.median(fraud_scores)),
        "threshold": float(threshold),
    }])

    # Calculate Population Stability Index (PSI) for each feature.
    # PSI is a descriptive drift indicator, not proof of model deterioration.
    drift_rows = []

    for feature in reference.columns:
        ref_values = reference[feature].to_numpy(dtype=float)
        new_values = incoming[feature].to_numpy(dtype=float)

        # Use reference-data quantiles to define comparison bins.
        bin_edges = np.unique(
            np.quantile(ref_values, np.linspace(0, 1, 11))
        )

        if len(bin_edges) < 3:
            drift_rows.append({
                "feature": feature,
                "psi": np.nan,
                "warning": "Insufficient distinct reference values",
            })
            continue

        # Extend the outer bins so incoming extreme values are included.
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        ref_counts, _ = np.histogram(ref_values, bins=bin_edges)
        new_counts, _ = np.histogram(new_values, bins=bin_edges)

        # Small smoothing prevents division by zero.
        ref_pct = ref_counts / len(ref_values)
        new_pct = new_counts / len(new_values)

        ref_pct = np.clip(ref_pct, 1e-6, None)
        new_pct = np.clip(new_pct, 1e-6, None)

        psi = float(np.sum(
            (new_pct - ref_pct) * np.log(new_pct / ref_pct)
        ))

        drift_rows.append({
            "feature": feature,
            "psi": psi,
            "warning": (
                "Investigate feature drift"
                if psi >= psi_warning_threshold
                else "No warning"
            ),
        })

    drift_report = pd.DataFrame(drift_rows)

    # Save monitoring outputs.
    prediction_log.to_csv(
        output_dir / "prediction_log.csv", index=False
    )
    batch_summary.to_csv(
        output_dir / "batch_summary.csv", index=False
    )
    drift_report.to_csv(
        output_dir / "feature_drift_report.csv", index=False
    )

    return prediction_log, batch_summary, drift_report
