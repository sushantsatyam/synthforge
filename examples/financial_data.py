"""Example: Financial transaction data with MNPI detection.

Demonstrates how SynthForge handles financial data with potential
material non-public information and generates safe synthetic alternatives.

To run with LLM-powered MNPI detection:
    export ANTHROPIC_API_KEY=your-key
    python examples/financial_data.py --llm
"""

import argparse
import numpy as np
import pandas as pd

from synthforge import SynthForge
from synthforge.constraints import PositiveValue, Inequality, ValueRange


def create_financial_sample(n: int = 2500) -> pd.DataFrame:
    """Simulate a financial transaction table from a data warehouse."""
    np.random.seed(42)

    dates = pd.date_range("2023-01-01", "2024-12-31", periods=n)
    revenue = np.random.lognormal(12, 1.5, n).round(2)  # Wide range
    costs = revenue * np.random.uniform(0.4, 0.85, n)

    return pd.DataFrame({
        "transaction_id": [f"TXN-{i:06d}" for i in range(n)],
        "trade_date": dates,
        "settlement_date": dates + pd.Timedelta(days=2),
        "client_name": [f"Client_{np.random.randint(1, 200)}" for _ in range(n)],
        "ticker": np.random.choice(["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "META"], n),
        "side": np.random.choice(["BUY", "SELL"], n, p=[0.55, 0.45]),
        "quantity": np.random.randint(10, 10000, n),
        "price": np.random.lognormal(4.5, 0.8, n).round(2),
        "notional": np.random.lognormal(10, 1.2, n).round(2),
        "revenue": revenue,
        "cost": costs.round(2),
        "profit_margin": ((revenue - costs) / revenue * 100).round(2),
        "desk": np.random.choice(["Equities", "Fixed Income", "Derivatives", "FX"], n),
        "trader_id": [f"TR-{np.random.randint(100, 150)}" for _ in range(n)],
        "status": np.random.choice(["Settled", "Pending", "Failed"], n, p=[0.85, 0.12, 0.03]),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="Enable LLM-based MNPI detection")
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    print("=" * 70)
    print("SynthForge Example: Financial Transaction Data")
    print("=" * 70)

    df = create_financial_sample(2500)
    print(f"\nInput: {len(df)} rows x {len(df.columns)} columns")
    print(f"\nSample:\n{df.head(3).to_string()}\n")

    # Initialize with or without LLM
    kwargs = {}
    if args.llm:
        kwargs["llm_provider"] = args.provider
        kwargs["llm_model"] = args.model
        print("LLM-powered MNPI detection ENABLED")
    else:
        print("Running without LLM (heuristic-only detection)")

    forge = SynthForge(verbose=True, **kwargs)

    # Add financial constraints
    forge.add_constraint(PositiveValue("quantity"))
    forge.add_constraint(PositiveValue("price"))
    forge.add_constraint(PositiveValue("notional"))
    forge.add_constraint(Inequality("cost", "revenue"))
    forge.add_constraint(ValueRange("profit_margin", low=-50, high=100))

    # Enable MNPI detection
    forge.config.privacy.detect_mnpi = args.llm

    # Profile
    meta = forge.profile(df, primary_key="transaction_id")
    print(f"\nPII columns: {meta.pii_columns}")
    print(f"MNPI columns: {meta.mnpi_columns}")

    # Generate
    forge.fit(df, primary_key="transaction_id")
    syn = forge.generate(num_rows=10_000)
    print(f"\nGenerated {len(syn)} rows")
    print(f"\nSynthetic sample:\n{syn.head(3).to_string()}\n")

    # Evaluate
    report = forge.evaluate(df, syn)
    print(f"\n{report.summary()}")


if __name__ == "__main__":
    main()
