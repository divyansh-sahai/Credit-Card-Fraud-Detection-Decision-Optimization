
-- Credit Card Fraud Detection & Decision Optimization Platform
-- Database schema for storing transaction predictions and outcomes.

CREATE TABLE IF NOT EXISTS transaction_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_time REAL NOT NULL,
    transaction_amount REAL NOT NULL CHECK (transaction_amount >= 0),
    fraud_probability REAL NOT NULL
        CHECK (fraud_probability >= 0 AND fraud_probability <= 1),
    decision_threshold REAL NOT NULL
        CHECK (decision_threshold >= 0 AND decision_threshold <= 1),
    predicted_fraud INTEGER NOT NULL
        CHECK (predicted_fraud IN (0, 1)),
    actual_fraud INTEGER
        CHECK (actual_fraud IN (0, 1) OR actual_fraud IS NULL),
    prediction_timestamp_utc TEXT NOT NULL,
    model_name TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_actual_fraud
    ON transaction_predictions (actual_fraud);

CREATE INDEX IF NOT EXISTS idx_predictions_predicted_fraud
    ON transaction_predictions (predicted_fraud);

CREATE INDEX IF NOT EXISTS idx_predictions_transaction_time
    ON transaction_predictions (transaction_time);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
    ON transaction_predictions (prediction_timestamp_utc);
