"""Example: E-commerce product catalog — heavily categorical data with
class imbalance and hierarchical categories."""

import numpy as np
import pandas as pd

from synthforge import SynthForge
from synthforge.constraints import PositiveValue, ValueRange


def create_ecommerce_data(n: int = 2500) -> pd.DataFrame:
    """Simulate an e-commerce product catalog with hierarchical categories."""
    np.random.seed(42)

    categories = {
        "Electronics": ["Phones", "Laptops", "Tablets", "Accessories"],
        "Clothing": ["Shirts", "Pants", "Dresses", "Shoes"],
        "Home": ["Furniture", "Kitchen", "Decor", "Lighting"],
        "Sports": ["Fitness", "Outdoor", "Team Sports", "Cycling"],
    }
    main_cats = list(categories.keys())
    main_probs = [0.35, 0.30, 0.20, 0.15]

    rows = []
    for i in range(n):
        cat = np.random.choice(main_cats, p=main_probs)
        subcat = np.random.choice(categories[cat])
        price = max(5, np.random.lognormal(3.5, 1.2))
        discount = np.random.choice([0, 5, 10, 15, 20, 25, 30], p=[0.4, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05])
        rows.append({
            "product_id": f"PRD-{i+1:05d}",
            "category": cat,
            "subcategory": subcat,
            "brand": np.random.choice(
                ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE", "Other"],
                p=[0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
            ),
            "price": round(price, 2),
            "discount_pct": discount,
            "sale_price": round(price * (1 - discount / 100), 2),
            "rating": round(np.random.beta(5, 2) * 5, 1),
            "review_count": int(np.random.exponential(50)),
            "in_stock": np.random.choice([True, False], p=[0.85, 0.15]),
            "condition": np.random.choice(["New", "Refurbished", "Open Box"], p=[0.80, 0.12, 0.08]),
            "warehouse": np.random.choice(["East", "West", "Central"], p=[0.4, 0.35, 0.25]),
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("SynthForge Example: E-commerce Catalog (Categorical-Heavy)")
    print("=" * 70)

    df = create_ecommerce_data(2500)
    print(f"\nInput: {len(df)} rows x {len(df.columns)} columns")
    print(f"\nCategory distribution (real):")
    print(df["category"].value_counts(normalize=True).round(3).to_string())

    forge = SynthForge(verbose=True)
    forge.add_constraint(PositiveValue("price"))
    forge.add_constraint(ValueRange("discount_pct", low=0, high=50))
    forge.add_constraint(ValueRange("rating", low=0, high=5))

    syn = forge.fit_generate(df, num_rows=10_000, primary_key="product_id")

    print(f"\nCategory distribution (synthetic):")
    print(syn["category"].value_counts(normalize=True).round(3).to_string())

    print(f"\nBrand distribution comparison:")
    for brand in ["BrandA", "BrandB", "BrandC"]:
        real_pct = (df["brand"] == brand).mean()
        syn_pct = (syn["brand"] == brand).mean()
        print(f"  {brand}: real={real_pct:.3f}, synth={syn_pct:.3f}")

    report = forge.evaluate(df, syn)
    print(f"\n{report.summary()}")


if __name__ == "__main__":
    main()
