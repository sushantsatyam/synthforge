"""GReaT synthesizer — Generation of Realistic Tabular data via LLM fine-tuning.

Implements the GReaT approach (Borisov et al., ICLR 2023):
  - Serialize each row as natural language: "Age is 34, Salary is 72000, ..."
  - Fine-tune a causal LM (GPT-2/DistilGPT-2) on these text representations
  - Sample by prompting with partial rows and letting the LM complete them
  - Random column order permutation during training enables arbitrary conditioning

Key advantages:
  - Handles mixed types naturally (everything is text)
  - Leverages pretrained semantic knowledge (understands "age", "salary" semantics)
  - Supports conditional generation without retraining
  - Captures complex inter-column dependencies via attention

Key limitations:
  - Very slow training (hours vs minutes for CTGAN)
  - Numerical precision limited by tokenization
  - Requires HuggingFace transformers library

Paper: https://arxiv.org/abs/2210.06280
Package: pip install be-great (reference implementation)

Requires: pip install "synthforge[llm]" + transformers
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import pandas as pd

from synthforge.synthesizers import BaseSynthesizer, register_synthesizer

logger = logging.getLogger(__name__)


def _check_deps():
    """Check that transformers + torch are available."""
    try:
        import torch
        import transformers
        return torch, transformers
    except ImportError as e:
        raise ImportError(
            "GReaT requires PyTorch + HuggingFace Transformers. "
            "Install with: pip install torch transformers"
        ) from e


def _require_cuda():
    torch, transformers = _check_deps()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GReaT requires CUDA GPU. LLM fine-tuning is extremely slow on CPU. "
            "Use synthesizer='gaussian_copula' for CPU workloads."
        )
    return torch, transformers


def _row_to_text(row: dict, columns: list[str], permute: bool = True) -> str:
    """Serialize a DataFrame row as natural language text.

    Format: "ColumnA is ValueA, ColumnB is ValueB, ..."
    Random column permutation during training helps the model learn all orderings.
    """
    cols = list(columns)
    if permute:
        random.shuffle(cols)
    parts = []
    for col in cols:
        val = row.get(col, "")
        if pd.isna(val):
            val = "missing"
        parts.append(f"{col} is {val}")
    return ", ".join(parts)


def _text_to_row(text: str, columns: list[str]) -> dict | None:
    """Parse generated text back into a row dictionary."""
    row = {}
    parts = text.split(",")
    for part in parts:
        part = part.strip()
        if " is " in part:
            key, value = part.split(" is ", 1)
            key = key.strip()
            value = value.strip()
            if key in columns:
                row[key] = value
    # Only return if we got most columns
    if len(row) >= len(columns) * 0.5:
        return row
    return None


@register_synthesizer("great")
class GReaTSynthesizer(BaseSynthesizer):
    """GReaT: LLM-based tabular data generation via fine-tuning.

    Trains a causal language model to generate rows as text, then
    parses the generated text back into structured tabular format.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 5e-5,
        max_length: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        cuda: bool = True,
        random_state: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._max_length = max_length
        self._temperature = temperature
        self._top_k = top_k
        self._cuda = cuda
        self._random_state = random_state

        self._model = None
        self._tokenizer = None
        self._device = None
        self._original_columns: list[str] = []
        self._column_dtypes: dict[str, str] = {}

    def fit(self, data: np.ndarray, column_names: list[str] | None = None) -> None:
        """Fine-tune LM on text-serialized table rows.

        Note: GReaT operates on the original DataFrame, not the transformed
        numerical array. But we conform to the BaseSynthesizer API by accepting
        numpy arrays + column_names and reconstructing the text internally.
        """
        if self._cuda:
            torch, transformers = _require_cuda()
        else:
            torch, transformers = _check_deps()
            logger.warning("GReaT on CPU — very slow fine-tuning. Expect hours.")

        n_samples, n_cols = data.shape
        self._n_columns = n_cols
        self._column_names = column_names or [f"col_{i}" for i in range(n_cols)]
        self._original_columns = list(self._column_names)
        self._device = torch.device("cuda" if self._cuda and torch.cuda.is_available() else "cpu")

        if self._random_state is not None:
            torch.manual_seed(self._random_state)
            random.seed(self._random_state)

        logger.info(
            "Fitting GReaT: %d samples, %d cols, model=%s, device=%s, epochs=%d",
            n_samples, n_cols, self._model_name, self._device, self._epochs,
        )

        # Load pretrained model + tokenizer
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._model_name)
        self._model = transformers.AutoModelForCausalLM.from_pretrained(self._model_name)
        self._model = self._model.to(self._device)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Convert numpy array back to text rows
        df = pd.DataFrame(data, columns=self._column_names)
        texts = []
        for _, row in df.iterrows():
            text = _row_to_text(row.to_dict(), self._original_columns, permute=True)
            texts.append(text)

        # Tokenize
        encodings = self._tokenizer(
            texts,
            truncation=True,
            max_length=self._max_length,
            padding=True,
            return_tensors="pt",
        )

        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self._batch_size, shuffle=True
        )

        # Fine-tune with causal LM objective
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._lr)
        self._model.train()

        for epoch in range(self._epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_ids, batch_mask in loader:
                batch_ids = batch_ids.to(self._device)
                batch_mask = batch_mask.to(self._device)

                outputs = self._model(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                    labels=batch_ids,
                )
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "GReaT Epoch %d/%d — Loss: %.4f",
                    epoch + 1, self._epochs, epoch_loss / max(n_batches, 1),
                )

        self._fitted = True
        logger.info("GReaT fine-tuning complete")

    def sample(self, n_rows: int) -> np.ndarray:
        """Generate rows by prompting the fine-tuned LM."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")

        torch, _ = _check_deps()
        logger.info("Sampling %d rows from GReaT (LLM generation)", n_rows)

        self._model.eval()
        generated_rows = []
        attempts = 0
        max_attempts = n_rows * 5  # Allow retries for failed parses

        with torch.no_grad():
            while len(generated_rows) < n_rows and attempts < max_attempts:
                batch = min(self._batch_size, n_rows - len(generated_rows))

                # Start with a random column as prompt
                start_col = random.choice(self._original_columns)
                prompt = f"{start_col} is"

                input_ids = self._tokenizer(
                    [prompt] * batch,
                    return_tensors="pt",
                    padding=True,
                ).input_ids.to(self._device)

                outputs = self._model.generate(
                    input_ids,
                    max_new_tokens=self._max_length,
                    temperature=self._temperature,
                    top_k=self._top_k,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

                for seq in outputs:
                    text = self._tokenizer.decode(seq, skip_special_tokens=True)
                    row = _text_to_row(text, self._original_columns)
                    if row is not None:
                        generated_rows.append(row)
                    attempts += 1

        # Convert to DataFrame then numpy
        if not generated_rows:
            logger.warning("GReaT failed to generate valid rows. Returning zeros.")
            return np.zeros((n_rows, self._n_columns))

        df = pd.DataFrame(generated_rows[:n_rows], columns=self._original_columns)

        # Try to convert back to numeric
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

        # Fill any remaining NaN
        result = df.apply(pd.to_numeric, errors="coerce").fillna(0).values
        return result[:n_rows]
