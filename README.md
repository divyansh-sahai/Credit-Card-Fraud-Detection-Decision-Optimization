
# Credit Card Fraud Detection & Decision Optimization Platform

An end-to-end machine learning and financial risk analytics project that detects fraudulent credit card transactions, optimizes fraud detection decisions using cost-sensitive analysis, serves predictions through a FastAPI service, and monitors model behaviour through statistical drift analysis and SQL-based reporting.


## Author

I am Divyansh Sahai, a postgraduate in Economics from the Indian Institute of Foreign Trade (IIFT), with a strong interest in financial analytics, risk management, and data science. My work focuses on applying Python, SQL, statistical analysis, and machine learning to financial and business problems, including fraud detection, credit risk modelling, financial valuation, and market risk analysis.

## Business Problem and Objectives

Credit card fraud detection presents a significant challenge because fraudulent transactions are extremely rare compared with legitimate transactions. A model must identify suspicious activity while controlling false positives that can lead to unnecessary transaction reviews or declines. This project develops a fraud classification system, evaluates competing machine learning models, optimizes decision thresholds under assumed cost scenarios, and builds supporting API, monitoring, and SQL analytics workflows.

## Key Features

- **Exploratory Data Analysis:** Investigates class imbalance, transaction amounts, temporal patterns, and anonymized features.
- **Machine Learning:** Compares Logistic Regression, Random Forest, and XGBoost against a dummy baseline.
- **Decision Optimization:** Evaluates fraud detection thresholds using precision, recall, false positives, missed fraud, and illustrative decision costs.
- **Prediction API:** Serves fraud probabilities and threshold-based classifications through FastAPI.
- **Model Monitoring:** Tracks prediction activity, fraud-score distributions, and feature drift using Population Stability Index (PSI).
- **SQL Analytics:** Uses SQLite and SQL queries to analyze classification outcomes, transaction amount bands, and estimated decision costs.
- **Automated Testing:** Uses Pytest to validate API endpoints and invalid-input handling.

## Technology Stack

- **Programming and Data Analysis:** Python, Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Visualization:** Matplotlib, Seaborn
- **API Development:** FastAPI, Uvicorn, Pydantic
- **Database and Analytics:** SQL, SQLite
- **Testing:** Pytest
- **Development Environment:** Anaconda, Jupyter Notebook
- **Version Control and Publication:** GitHub


## Dataset Description and Limitations

This project uses the public Credit Card Fraud Detection dataset from Kaggle, containing 284,807 transactions and 492 fraudulent cases. The dataset includes `Time`, `Amount`, 28 anonymized PCA-transformed features (`V1`–`V28`), and a binary target variable (`Class`). Fraud represents approximately 0.173% of all transactions, creating a highly imbalanced classification problem.

The dataset covers approximately two days of transactions from September 2013. Most features are anonymized, limiting direct business interpretation, and the short observation window restricts conclusions about long-term patterns. The raw dataset is excluded from the repository due to its size; access instructions are provided below.

## Project Architecture

The workflow follows an end-to-end structure:

1. **Data Analysis:** Audit the dataset, investigate distributions, and prepare chronological training, validation, and test sets.
2. **Model Development:** Train and compare baseline and machine learning classifiers.
3. **Decision Optimization:** Select a decision threshold using validation data and evaluate the finalized model on the held-out test set.
4. **Prediction Service:** Load the saved model through FastAPI and return fraud probabilities and classifications.
5. **Monitoring:** Analyze prediction activity, feature distribution shifts, and labelled performance.
6. **SQL Analytics:** Store prediction records in SQLite and generate analytical reports using SQL queries.

## Repository Structure

```text
credit-card-fraud-detection/
├── api/                  # FastAPI service and API documentation
├── data/                 # Local dataset (excluded from GitHub)
├── models/               # Saved model artifact (excluded from GitHub)
├── notebooks/            # Data audit, analysis, and modelling notebook
├── reports/
│   ├── figures/          # Exploratory analysis and evaluation charts
│   ├── monitoring/       # Monitoring reports
│   ├── sql/              # SQL query results
│   └── cost_sensitivity_analysis.csv
├── sql/                  # Database schema and analytics queries
├── src/                  # Reusable monitoring module
├── tests/                # Automated API tests
├── .gitignore
├── requirements.txt
└── README.md
```

## Data Preparation and Exploratory Analysis

The dataset was audited for missing values, infinite values, unexpected target labels, invalid transaction amounts, and duplicate records. Duplicate rows were investigated and retained rather than automatically removed.

The data was divided chronologically into training (199,364 transactions), validation (42,721 transactions), and test (42,722 transactions). These sets contained 384, 56, and 52 fraud cases, respectively. Exploratory analysis examined class imbalance, transaction amounts, temporal patterns, and differences in anonymized features. Visualizations were saved in `reports/figures/`.

The chronological split reduces the risk of using future observations to train a model evaluated on earlier observations. However, the dataset's short time span remains a limitation.
``

## Model Development and Selection

Four classification approaches were evaluated: a Dummy Classifier baseline, Logistic Regression, Random Forest, and XGBoost. Model comparison used ROC-AUC, Precision-Recall AUC (PR-AUC), precision, recall, and F1-score, with particular attention to the severe class imbalance.

Random Forest was selected for the final decision workflow based on the validation results and the project's decision-making objectives. The decision threshold was subsequently assessed using validation data rather than selected using the final test results.

## Model Evaluation and Results

The selected Random Forest model was evaluated on the held-out test set of 42,722 transactions, including 52 actual fraud cases. At the selected threshold of 0.10, the model achieved the following results:

| Metric | Test Result |
|---|---:|
| ROC-AUC | 0.9523 |
| PR-AUC | 0.7779 |
| Precision | 46.99% |
| Recall | 75.00% |
| F1-score | 57.78% |
| True positives | 39 |
| False positives | 44 |
| False negatives | 13 |
| True negatives | 42,626 |

The model detected 39 of the 52 fraud cases while incorrectly flagging 44 legitimate transactions. These results illustrate the trade-off between fraud detection and false-positive control in an imbalanced classification problem.

## Fraud Decision Optimization and Cost Analysis

The decision threshold was optimized using validation data under different assumed costs for missed fraud and false positives. Sensitivity analysis examined missed-fraud cost weights of 5, 10, 20, and 50 units, with a false-positive cost of 1 unit.

The final test-set results at threshold 0.10 included 13 missed fraud cases and 44 false positives. Under the illustrative cost assumptions of 10 units per missed fraud case and 1 unit per false positive, the estimated decision cost was 174 units.

These values are simulated cost estimates, not actual monetary losses, recovered funds, or realized business savings. The analysis demonstrates how different risk preferences can influence threshold selection.

## Model Monitoring and Drift Analysis

A reusable Python monitoring module was developed to summarize incoming prediction batches, record fraud scores and decisions, and compare feature distributions against a reference dataset using the Population Stability Index (PSI).

A controlled batch of 500 test transactions produced a 0% flagged-transaction rate. Several features generated PSI warnings, with `Time` showing the largest distribution shift. Because the dataset is divided chronologically and covers only a short period, these shifts must be interpreted in that context.

The monitoring workflow also includes labelled performance evaluation and saved reports covering batch summaries, prediction logs, feature drift, and test-set performance. These experiments demonstrate offline monitoring capabilities; they do not establish live production drift or ongoing model deterioration.

## SQL Analytics and Reporting

SQLite was used to store 42,722 test-set transaction prediction records, including fraud probabilities, decision thresholds, predicted classifications, and known outcomes. Five SQL queries were developed to analyze overall classification performance, actual versus predicted fraud, transaction amount bands, illustrative decision costs, and fraud-score distributions.

The resulting CSV reports are saved in `reports/sql/`. The database schema and analytical queries are available in `sql/schema.sql` and `sql/fraud_analytics.sql`. The database and reports support reproducible analysis of the model's decisions and their assumed cost implications.

## FastAPI Prediction Service

A FastAPI application serves predictions using the saved Random Forest model. The service exposes three endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /` | Returns API status and model information |
| `GET /health` | Reports service health and the decision threshold |
| `POST /predict` | Accepts transaction features and returns a fraud probability, classification, and threshold |

The API validates required features and rejects missing or unexpected features, as well as negative transaction amounts or times. Interactive API documentation is available through FastAPI's `/docs` endpoint when the service is running locally.

## Installation and Setup

The project was developed using Python through Anaconda and a dedicated Conda environment named `fraud_detection`. The environment includes the libraries needed for data analysis, machine learning, API development, and testing.

Clone or download the repository, then open a terminal in the project directory. Create and activate a Python environment, and install the dependencies listed in `requirements.txt`.

The raw Kaggle dataset and saved model artifact are excluded from GitHub. Download the dataset separately and place `creditcard.csv` in the `data/` directory. The trained model must be generated or supplied locally before running the API.

## How to Run the Project

1. Open `notebooks/01_data_audit.ipynb` in Jupyter using an environment with the required dependencies installed.
2. Run the notebook cells in order to reproduce the analysis and modelling workflow, subject to the available notebook variables and saved artifacts.
3. Ensure the trained model artifact exists at `models/fraud_random_forest.joblib`.
4. From the project root, start the API with:

   ```bash
   python -m uvicorn api.fraud_detection_api:app --reload
   ```

5. Open `http://127.0.0.1:8000/docs` to inspect and test the API.
6. Use the SQL scripts in `sql/` to create the database schema and run the analytical queries after preparing the database records.

## Automated Testing

The API was tested using Pytest and FastAPI's test client. Five automated tests passed, covering the home endpoint, health endpoint, valid prediction requests, missing-feature rejection, and negative transaction amount rejection.

Run the tests from the project root with:

```bash
python -m pytest tests/test_fraud_detection_api.py -v
```

## Key Findings and Business Implications

- The dataset is highly imbalanced, with fraud accounting for approximately 0.173% of transactions.
- The selected Random Forest model achieved a test ROC-AUC of 0.9523 and PR-AUC of 0.7779.
- At the selected threshold of 0.10, the model detected 39 of 52 fraud cases, with 44 false positives and 13 missed fraud cases.
- The illustrative decision cost was 174 units under the assumed cost weights.
- The monitoring and SQL workflows demonstrate how prediction activity, feature distribution changes, classification outcomes, and decision costs can be analyzed systematically.

These findings illustrate the practical trade-offs involved in fraud detection. They do not establish actual financial savings or production performance.

## Limitations and Future Improvements

The dataset covers only approximately two days of historical transactions, and most predictors are anonymized PCA-transformed features. The model's performance may therefore differ on new datasets or in a live transaction environment.

The monitoring workflow uses controlled offline batches rather than a continuous production feed. The financial decision framework relies on assumed cost weights rather than observed chargeback losses or customer-retention costs.

Future improvements could include evaluation on newer datasets, longer-term drift monitoring, more detailed cost modelling, explainability analysis, and deployment in a properly secured hosted environment.

## Reproducibility and Data Access

The project uses the public Credit Card Fraud Detection dataset available on Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download the dataset and place `creditcard.csv` inside the project's `data/` directory. The raw dataset, trained model artifact, and generated SQLite database are excluded from the repository and must be recreated or obtained separately.

The repository includes the notebook, API source code, automated tests, monitoring module, SQL scripts, and analytical reports to document the workflow and support reproduction of the analysis.
