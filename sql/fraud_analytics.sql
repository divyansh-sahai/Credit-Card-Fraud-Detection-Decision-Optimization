
-- Credit Card Fraud Detection & Decision Optimization Platform
-- SQL analytics using the transaction_predictions table.

-- Query 1: Overall fraud detection performance.
SELECT
    COUNT(*) AS total_transactions,
    SUM(actual_fraud) AS actual_fraud_cases,
    SUM(predicted_fraud) AS flagged_transactions,
    SUM(CASE
        WHEN actual_fraud = 1 AND predicted_fraud = 1
        THEN 1 ELSE 0
    END) AS true_positives,
    SUM(CASE
        WHEN actual_fraud = 0 AND predicted_fraud = 1
        THEN 1 ELSE 0
    END) AS false_positives,
    SUM(CASE
        WHEN actual_fraud = 1 AND predicted_fraud = 0
        THEN 1 ELSE 0
    END) AS false_negatives,
    ROUND(
        1.0 * SUM(CASE
            WHEN actual_fraud = 1 AND predicted_fraud = 1
            THEN 1 ELSE 0
        END)
        / NULLIF(SUM(predicted_fraud), 0),
        4
    ) AS precision,
    ROUND(
        1.0 * SUM(CASE
            WHEN actual_fraud = 1 AND predicted_fraud = 1
            THEN 1 ELSE 0
        END)
        / NULLIF(SUM(actual_fraud), 0),
        4
    ) AS recall
FROM transaction_predictions
WHERE actual_fraud IS NOT NULL;


-- Query 2: Compare actual and predicted fraud counts.
SELECT
    actual_fraud,
    predicted_fraud,
    COUNT(*) AS transaction_count
FROM transaction_predictions
WHERE actual_fraud IS NOT NULL
GROUP BY actual_fraud, predicted_fraud
ORDER BY actual_fraud, predicted_fraud;


-- Query 3: Performance by transaction amount band.
SELECT
    CASE
        WHEN transaction_amount < 25 THEN 'Under 25'
        WHEN transaction_amount < 100 THEN '25 to under 100'
        WHEN transaction_amount < 500 THEN '100 to under 500'
        ELSE '500 and above'
    END AS amount_band,
    COUNT(*) AS total_transactions,
    SUM(actual_fraud) AS actual_fraud_cases,
    SUM(predicted_fraud) AS flagged_transactions,
    SUM(CASE
        WHEN actual_fraud = 1 AND predicted_fraud = 1
        THEN 1 ELSE 0
    END) AS true_positives,
    SUM(CASE
        WHEN actual_fraud = 0 AND predicted_fraud = 1
        THEN 1 ELSE 0
    END) AS false_positives,
    SUM(CASE
        WHEN actual_fraud = 1 AND predicted_fraud = 0
        THEN 1 ELSE 0
    END) AS false_negatives
FROM transaction_predictions
WHERE actual_fraud IS NOT NULL
GROUP BY amount_band
ORDER BY
    CASE amount_band
        WHEN 'Under 25' THEN 1
        WHEN '25 to under 100' THEN 2
        WHEN '100 to under 500' THEN 3
        ELSE 4
    END;


-- Query 4: Estimated decision costs under explicit assumptions.
-- These are illustrative units, NOT actual monetary losses.
-- Assumption: missed fraud costs 10 units; false positive costs 1 unit.
SELECT
    SUM(CASE
        WHEN actual_fraud = 1 AND predicted_fraud = 0
        THEN 1 ELSE 0
    END) AS false_negatives,
    SUM(CASE
        WHEN actual_fraud = 0 AND predicted_fraud = 1
        THEN 1 ELSE 0
    END) AS false_positives,
    10.0 * SUM(CASE
        WHEN actual_fraud = 1 AND predicted_fraud = 0
        THEN 1 ELSE 0
    END)
    + 1.0 * SUM(CASE
        WHEN actual_fraud = 0 AND predicted_fraud = 1
        THEN 1 ELSE 0
    END) AS estimated_decision_cost_units
FROM transaction_predictions
WHERE actual_fraud IS NOT NULL;


-- Query 5: Fraud-score distribution by actual outcome.
SELECT
    actual_fraud,
    COUNT(*) AS transaction_count,
    ROUND(AVG(fraud_probability), 6) AS mean_fraud_probability,
    ROUND(MIN(fraud_probability), 6) AS minimum_fraud_probability,
    ROUND(MAX(fraud_probability), 6) AS maximum_fraud_probability
FROM transaction_predictions
WHERE actual_fraud IS NOT NULL
GROUP BY actual_fraud
ORDER BY actual_fraud;
