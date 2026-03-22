"""SynthForge CLI."""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd


def main():
    p = argparse.ArgumentParser(prog="synthforge", description="Generate synthetic tabular data")
    p.add_argument("input", help="Input CSV/Parquet path")
    p.add_argument("-o", "--output", default="synthetic_output.csv")
    p.add_argument("-n", "--num-rows", type=int, default=None)
    p.add_argument("-s", "--synthesizer", default="auto", choices=["auto","gaussian_copula","ctgan","tvae"])
    p.add_argument("--primary-key", default=None)
    p.add_argument("--llm-provider", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--report", default=None)
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows x {len(df.columns)} cols")

    from synthforge import SynthForge
    from synthforge.config import SynthForgeConfig, GenerationConfig

    config = SynthForgeConfig(generation=GenerationConfig(seed=args.seed) if args.seed else GenerationConfig())
    forge = SynthForge(config=config, synthesizer=args.synthesizer, llm_provider=args.llm_provider,
                       llm_model=args.llm_model, verbose=not args.quiet)

    synthetic_df = forge.fit_generate(df, num_rows=args.num_rows or len(df), primary_key=args.primary_key)

    if args.output.endswith(".parquet"):
        synthetic_df.to_parquet(args.output, index=False)
    else:
        synthetic_df.to_csv(args.output, index=False)
    print(f"Saved {len(synthetic_df)} rows to {args.output}")

    if args.evaluate:
        report = forge.evaluate(df, synthetic_df)
        print("\n" + report.summary())
        if args.report:
            with open(args.report, "w") as f:
                json.dump(report.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
