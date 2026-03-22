"""Example: Full SynthForge pipeline on a sample HR dataset."""

import numpy as np
import pandas as pd

from synthforge import SynthForge
from synthforge.constraints import Inequality, PositiveValue


def create_sample_hr_data(n: int = 2500) -> pd.DataFrame:
    """Simulate a production HR table sample from Redshift."""
    np.random.seed(42)
    departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]
    levels = ["Junior", "Mid", "Senior", "Lead", "Director", "VP"]

    hire_dates = pd.date_range("2010-01-01", "2024-06-01", periods=n)
    ages = np.random.randint(22, 65, n)
    base_salary = np.random.lognormal(10.8, 0.5, n).round(2)

    df = pd.DataFrame({
        "employee_id": range(10001, 10001 + n),
        "first_name": [f"FirstName_{i}" for i in range(n)],
        "last_name": [f"LastName_{i}" for i in range(n)],
        "email": [f"employee{i}@company.com" for i in range(n)],
        "phone": [f"+1-555-{np.random.randint(100,999)}-{np.random.randint(1000,9999)}" for _ in range(n)],
        "age": ages,
        "department": np.random.choice(departments, n),
        "level": np.random.choice(levels, n, p=[0.3, 0.25, 0.2, 0.12, 0.08, 0.05]),
        "base_salary": base_salary,
        "bonus_pct": np.random.uniform(0.05, 0.30, n).round(3),
        "total_comp": base_salary * (1 + np.random.uniform(0.05, 0.30, n)),
        "hire_date": hire_dates,
        "is_remote": np.random.choice([True, False], n, p=[0.35, 0.65]),
        "performance_score": np.random.normal(3.5, 0.8, n).clip(1, 5).round(1),
        "satisfaction": np.random.choice(["Very Low", "Low", "Medium", "High", "Very High"], n,
                                          p=[0.05, 0.10, 0.30, 0.35, 0.20]),
    })

    # Add some NULLs
    null_indices = np.random.choice(n, size=int(n * 0.05), replace=False)
    df.loc[null_indices, "phone"] = np.nan
    df.loc[np.random.choice(n, 30, replace=False), "bonus_pct"] = np.nan

    return df


def main():
    print("=" * 70)
    print("SynthForge Example: HR Dataset")
    print("=" * 70)

    # Step 1: Create sample data (simulating Redshift extract)
    df = create_sample_hr_data(2500)
    print(f"\nInput: {len(df)} rows x {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample:\n{df.head(3).to_string()}\n")

    # Step 2: Initialize SynthForge
    # Without LLM (pure statistical):
    forge = SynthForge(synthesizer="gaussian_copula", verbose=True)

    # With LLM enrichment (uncomment to use):
    # forge = SynthForge(
    #     llm_provider="anthropic",
    #     llm_model="claude-sonnet-4-20250514",
    #     llm_api_key="your-api-key-here",
    # )

    # Step 3: Add business constraints
    forge.add_constraint(PositiveValue("base_salary"))
    forge.add_constraint(Inequality("base_salary", "total_comp"))  # total >= base

    # Step 4: Profile the data
    metadata = forge.profile(df, primary_key="employee_id")
    print(f"\nMetadata:")
    print(f"  Strategy: {metadata.data_strategy}")
    print(f"  PII columns: {metadata.pii_columns}")
    print(f"  Numerical: {metadata.numerical_columns}")
    print(f"  Categorical: {metadata.categorical_columns}")

    # Step 5: Fit the model
    forge.fit(df, primary_key="employee_id")

    # Step 6: Generate synthetic data
    synthetic_df = forge.generate(num_rows=10_000)
    print(f"\nGenerated: {len(synthetic_df)} rows")
    print(f"\nSynthetic sample:\n{synthetic_df.head(3).to_string()}\n")

    # Step 7: Evaluate quality
    report = forge.evaluate(df, synthetic_df)
    print(f"\n{report.summary()}")

    # Step 8: Verify PII was replaced
    if metadata.pii_columns:
        print(f"\nPII Verification:")
        for col in metadata.pii_columns[:3]:
            print(f"  {col} — original: {df[col].iloc[0]}, synthetic: {synthetic_df[col].iloc[0]}")

    print("\nDone!")


if __name__ == "__main__":
    main()
