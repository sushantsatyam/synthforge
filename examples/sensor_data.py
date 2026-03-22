"""Example: Numerical sensor data — demonstrates high-quality copula synthesis
on correlated continuous features with no PII/categorical noise."""

import numpy as np
import pandas as pd

from synthforge import SynthForge


def create_sensor_data(n: int = 2500) -> pd.DataFrame:
    """Simulate correlated IoT sensor readings."""
    np.random.seed(42)
    # Correlated features: temperature drives humidity and pressure
    temp = np.random.normal(25, 8, n)
    humidity = 60 - 0.8 * temp + np.random.normal(0, 5, n)
    pressure = 1013 + 0.3 * temp + np.random.normal(0, 3, n)
    wind_speed = np.random.exponential(5, n)
    vibration = np.abs(np.random.normal(0, 0.5, n)) + wind_speed * 0.1

    return pd.DataFrame({
        "temperature_c": temp.round(2),
        "humidity_pct": humidity.clip(10, 100).round(1),
        "pressure_hpa": pressure.round(1),
        "wind_speed_ms": wind_speed.round(2),
        "vibration_g": vibration.round(4),
        "power_consumption_kw": (50 + 2 * temp + 0.5 * wind_speed + np.random.normal(0, 3, n)).round(2),
        "error_rate": np.random.beta(2, 50, n).round(5),
    })


def main():
    print("=" * 70)
    print("SynthForge Example: Numerical Sensor Data")
    print("=" * 70)

    df = create_sensor_data(2500)
    print(f"\nInput: {len(df)} rows x {len(df.columns)} columns")
    print(f"\nCorrelation matrix (real):")
    print(df.corr().round(3).to_string())

    forge = SynthForge(synthesizer="gaussian_copula", verbose=True)
    forge.fit(df)
    syn = forge.generate(10_000)

    print(f"\nCorrelation matrix (synthetic):")
    print(syn.corr().round(3).to_string())

    # Evaluate
    report = forge.evaluate(df, syn)
    print(f"\n{report.summary()}")

    # Show column-level comparison
    print("\n--- Column Statistics Comparison ---")
    for col in df.columns:
        print(f"  {col}:")
        print(f"    Real  — mean={df[col].mean():.3f}, std={df[col].std():.3f}, "
              f"min={df[col].min():.3f}, max={df[col].max():.3f}")
        print(f"    Synth — mean={syn[col].mean():.3f}, std={syn[col].std():.3f}, "
              f"min={syn[col].min():.3f}, max={syn[col].max():.3f}")


if __name__ == "__main__":
    main()
