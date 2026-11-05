# Task 1 — Data Cleaning & Preprocessing

**AI & ML Internship | Dataset: Titanic (Yasser H.)**

## Objective
Prepare raw passenger data for machine-learning use by exploring it, handling missing values, encoding categorical variables, standardizing numerical features, detecting outliers with boxplots, and removing selected outliers.

## Assignment requirements
This repository addresses all five technical steps in the assignment:
1. Import the dataset and inspect dimensions, data types, summary statistics, and null values.
2. Impute missing values.
3. Encode categorical variables.
4. Standardize numerical features.
5. Visualize and remove outliers using the IQR rule.

It also includes answers to all eight interview questions from the assignment.

## Dataset
The assignment specifies the [Yasser H. Titanic dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset). The CSV in `dataset/Titanic-Dataset.csv` contains 891 passenger records and the expected 12 columns. It is a public mirror of that dataset; the Kaggle page remains the canonical source.

## Project structure
```text
task-1-data-cleaning-preprocessing/
├── dataset/
│   └── Titanic-Dataset.csv
├── notebooks/
│   └── data_preprocessing.ipynb
├── src/
│   └── preprocess.py
├── outputs/                  # generated when the script/notebook runs
├── download_dataset.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the project
The dataset is already included. From the repository root:
```powershell
python src/preprocess.py
``

Or open `notebooks/data_preprocessing.ipynb` in VS Code/Jupyter and run all cells.

To download the dataset again from KaggleHub:
```powershell
python download_dataset.py
``

## Preprocessing decisions
- **Cabin:** dropped because it is very sparse in this dataset.
- **Age:** missing values are filled with the median age within each passenger class, then the overall median as a fallback.
- **Fare:** missing values are filled with the median.
- **Embarked:** missing values are filled with the mode.
- **Target:** rows with missing `Survived` are removed; the target is retained and never scaled.
- **Outliers:** boxplots are generated for `Age` and `Fare`. Rows outside the 1.5 × IQR fences for either column are removed for this introductory exercise.
- **Categorical encoding:** `Sex` and `Embarked` use one-hot encoding.
- **Scaling:** `Age` and `Fare` use `StandardScaler` (zero mean and unit variance).
- **Identifiers/free text:** `PassengerId`, `Name`, and `Ticket` are removed from the modeling table.

Outliers are not necessarily data errors. High fares can be valid observations, so removal should be evaluated with domain knowledge and model validation in a real project.

## Generated outputs
Running the script creates:
- `outputs/missing_values_before.csv`
- `outputs/missing_values_before.png`
- `outputs/boxplots_before_outlier_removal.png`
- `outputs/boxplots_after_outlier_removal.png`
- `outputs/age_distribution.png`
- `outputs/titanic_cleaned_encoded_scaled.csv`
- `outputs/preprocessing_summary.txt`

## Interview questions — answers

1. **What are the different types of missing data?**  
   MCAR means missingness is unrelated to any data; MAR means it depends on observed variables; MNAR means it depends on the missing value itself or an unobserved factor.

2. **How do you handle categorical variables?**  
   Use one-hot encoding for nominal categories and ordinal encoding only when categories have a genuine order. Fit encoders on training data only in predictive workflows.

3. **What is the difference between normalization and standardization?**  
   Normalization commonly rescales values to a fixed range such as [0, 1]. Standardization transforms values to approximately zero mean and unit standard deviation.

4. **How do you detect outliers?**  
   Use boxplots and the IQR rule, z-scores, robust statistics, and domain knowledge. An unusual value is not automatically an error.

5. **Why is preprocessing important in ML?**  
   It improves data quality, makes features usable by algorithms, handles missing/inconsistent values, and helps prevent leakage. Appropriate preprocessing can improve model stability and performance.

6. **What is one-hot encoding vs label encoding?**  
   One-hot encoding creates a binary column for each category. Label encoding maps categories to integer codes and may imply a false order when used on nominal categories.

7. **How do you handle data imbalance?**  
   Use stratified splitting, class weights, suitable metrics such as precision, recall, F1 and PR-AUC, and (when justified) resampling such as random oversampling or SMOTE on training data only.

8. **Can preprocessing affect model accuracy?**  
   Yes. Good imputation, encoding, scaling and leakage prevention can improve results; poor choices can remove useful information or introduce bias and reduce accuracy.

## Methodological note
This task demonstrates preprocessing; it does not train a survival classifier. In a real model evaluation, split the data first and fit imputers, encoders, scalers and outlier thresholds on training data only, preferably with a scikit-learn `Pipeline`.

## References
- Assignment dataset: https://www.kaggle.com/datasets/yasserh/titanic-dataset
- KaggleHub: https://github.com/Kaggle/kagglehub
- pandas: https://pandas.pydata.org/docs/
- scikit-learn preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html


## Cloud-generated results (2026-11-05)

The dataset was processed in a cloud runtime and the generated artifacts are committed under `outputs/`.

| Check | Result |
|---|---:|
| Original dataset | 891 rows × 12 columns |
| Rows after preprocessing | 755 |
| Rows removed by IQR rule | 136 |
| Missing values in final feature table | 0 |

### Generated artifacts
- [Cleaned, encoded and standardized CSV](outputs/titanic_cleaned_encoded_scaled.csv)
- [Execution report](outputs/execution_report.md)
- [Detailed preprocessing summary (JSON)](outputs/preprocessing_summary.json)
- [Missing-values chart (SVG)](outputs/missing_values_before.svg)
- [Boxplots before outlier removal (SVG)](outputs/boxplots_before_outlier_removal.svg)
- [Boxplots after outlier removal (SVG)](outputs/boxplots_after_outlier_removal.svg)
- [Standardized age distribution (SVG)](outputs/age_distribution.svg)

**Execution transparency:** The transformations and artifacts were generated in a cloud JavaScript runtime using the same CSV parsing, median imputation, linear IQR quantiles, one-hot encoding and population-standard-deviation scaling logic. The Python script and notebook are included as reproducible implementations, but the Python interpreter itself was not run in this environment. No ML model was trained.
