# Bank Term Deposit Subscription Predictor

An end-to-end classification project implementing a Decision Tree Classifier to predict client purchase behavior (term deposit subscriptions) for a retail bank's telemarketing campaigns.

## 📌 Project Overview
Direct marketing campaigns are resource-intensive. This project builds a predictive machine learning pipeline using demographic (age, job, marital status, education) and behavioral (balance, previous contact, loan history) data to target customers most likely to subscribe to a term deposit, optimizing outreach strategy and conversion rates.

## 🚀 Key Features
* **Machine Learning Pipeline:** Builds, trains, and evaluates a Decision Tree Classifier using Scikit-Learn.
* **Pre-modeling Data Processing:** Conducts data cleaning, handles outliers in financial balances, and performs categorical encoding (via `LabelEncoder`).
* **Feature Relationship Mapping:** Explores multicollinearity and feature interactions using heatmaps, box plots, and continuous distributions.
* **Decision Tree Visualization:** Plots the trained classification tree to demonstrate feature splits and model transparency.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Processing & EDA:** Pandas, NumPy, Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn (Decision Tree Classifier, Train-Test Split, Classification Metrics, LabelEncoder)

## 📈 Methodology & Pipeline
1. **Data Acquisition:** Sourced campaign history records from the [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).
2. **Exploratory Data Analysis (EDA):**
   * Plotted correlation heatmaps for numeric variables.
   * Investigated outlier profiles in variables like `balance` and `duration` using box plots.
   * Visualized class distribution of the target variable (`y`) using count plots.
3. **Data Preprocessing & Feature Engineering:**
   * Handled categorical text columns through label encoding to make them suitable for tree splits.
   * Split data into stratified training and testing subsets (typically an 80/20 partition).
4. **Model Architecture & Training:**
   * Instantiated and trained a `DecisionTreeClassifier` with regularized depth control to prevent overfitting.
5. **Model Evaluation:**
   * Measured prediction accuracy, precision, recall, and F1-score.
   * Rendered the graphical tree map to examine the critical decision-making nodes.

## 💡 Key Insights
* **Primary Predictor (Call Duration):** The duration of the last contact call (`duration`) is a powerful predictor of subscription success; longer calls correlate highly with conversion.
* **Outlier Dynamics:** Financial attributes like yearly average bank balances show high variance and extreme outliers, requiring robust split thresholds within the decision tree.
* **Model Explainability:** The visualized decision tree indicates that prior campaign success and contact duration form the initial classification boundaries.

## 📂 Project Structure
```text
├── Task_03.ipynb                           # Model building, evaluation, and EDA notebook
├── bank.csv                                 # UCI bank marketing campaign dataset
└── README.md                                # Project documentation
```

## 👤 Author
* **Khan Sohail**
  * [LinkedIn](https://www.linkedin.com/in/khan-sohail-386b2027a)
  * Email: ks646397@gmail.com
