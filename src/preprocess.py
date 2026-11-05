"""Data cleaning and preprocessing for AI & ML Internship Task 1."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run 'python download_dataset.py' "
            "or place Titanic-Dataset.csv in the dataset/ folder."
        )
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {"Survived", "Pclass", "Sex", "Age", "Fare", "Embarked"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    original_rows, original_columns = df.shape
    missing_before = df.isna().sum().sort_values(ascending=False)
    missing_before.to_csv(output_dir / "missing_values_before.csv", header=["missing_count"])

    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing_before.index, y=missing_before.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Missing values")
    plt.title("Missing values by column — before cleaning")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_values_before.png", dpi=160)
    plt.close()

    data = df.copy()
    dropped_columns = []
    if "Cabin" in data.columns:
        data = data.drop(columns="Cabin")
        dropped_columns.append("Cabin")

    age_medians = data.groupby("Pclass", observed=False)["Age"].transform("median")
    data["Age"] = data["Age"].fillna(age_medians).fillna(data["Age"].median())
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    if data["Embarked"].isna().any():
        data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode().iloc[0])
    data = data.dropna(subset=["Survived"]).copy()

    outlier_columns = ["Age", "Fare"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, outlier_columns):
        sns.boxplot(y=data[col], ax=ax)
        ax.set_title(f"{col} — before IQR filtering")
    plt.tight_layout()
    plt.savefig(output_dir / "boxplots_before_outlier_removal.png", dpi=160)
    plt.close()

    rows_before_filter = len(data)
    keep = pd.Series(True, index=data.index)
    bounds = {}
    for col in outlier_columns:
        q1, q3 = data[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        bounds[col] = {"lower": float(lower), "upper": float(upper)}
        keep &= data[col].between(lower, upper)
    data = data.loc[keep].copy()
    rows_removed = rows_before_filter - len(data)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, outlier_columns):
        sns.boxplot(y=data[col], ax=ax)
        ax.set_title(f"{col} — after IQR filtering")
    plt.tight_layout()
    plt.savefig(output_dir / "boxplots_after_outlier_removal.png", dpi=160)
    plt.close()

    for col in ["PassengerId", "Name", "Ticket"]:
        if col in data.columns:
            data = data.drop(columns=col)
            dropped_columns.append(col)

    data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=True, dtype=int)
    scale_columns = ["Age", "Fare"]
    scaler = StandardScaler()
    data[scale_columns] = scaler.fit_transform(data[scale_columns])

    output_csv = output_dir / "titanic_cleaned_encoded_scaled.csv"
    data.to_csv(output_csv, index=False)

    plt.figure(figsize=(8, 4))
    sns.histplot(data["Age"], bins=25, kde=True)
    plt.title("Standardized age distribution after preprocessing")
    plt.xlabel("Standardized Age")
    plt.tight_layout()
    plt.savefig(output_dir / "age_distribution.png", dpi=160)
    plt.close()

    summary = {
        "original_rows": int(original_rows),
        "original_columns": int(original_columns),
        "final_rows": int(len(data)),
        "final_columns": int(data.shape[1]),
        "rows_removed_as_outliers": int(rows_removed),
        "dropped_columns": dropped_columns,
        "scaled_columns": scale_columns,
        "remaining_missing_values": int(data.isna().sum().sum()),
        "outlier_bounds": bounds,
        "output_csv": str(output_csv),
    }
    (output_dir / "preprocessing_summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()),
        encoding="utf-8",
    )
    return data, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/Titanic-Dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    processed, summary = preprocess(load_dataset(args.input), args.output)
    print("Preprocessing complete.")
    print(pd.Series(summary).to_string())
    print("\nPreview:")
    print(processed.head())


if __name__ == "__main__":
    main()
