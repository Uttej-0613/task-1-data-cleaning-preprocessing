# Execution report

**Status:** Completed in cloud runtime against the committed Titanic CSV.

## Dataset and results
- Input: 891 rows × 12 columns
- Rows after IQR filtering: 755
- Rows removed as outliers: 136
- Missing values in final feature table: 0
- Categorical variables: Sex and Embarked one-hot encoded
- Standardized variables: Age and Fare

## Missing values before cleaning
- PassengerId: 0
- Survived: 0
- Pclass: 0
- Name: 0
- Sex: 0
- Age: 177
- SibSp: 0
- Parch: 0
- Ticket: 0
- Fare: 0
- Cabin: 687
- Embarked: 2

## Outlier bounds (1.5 × IQR)
- Age: [-0.5000, 59.5000]
- Fare: [-26.7240, 65.6344]

## Final feature columns
- Survived
- Pclass
- Age
- SibSp
- Parch
- Fare
- Sex_male
- Embarked_Q
- Embarked_S

## Important reproducibility note
The data transformations and outputs below were generated in the cloud runtime. The repository also includes a Python script and Jupyter notebook that implement the same workflow for reproducibility. This execution report is not a model-training or predictive-accuracy report; no classifier was trained.
