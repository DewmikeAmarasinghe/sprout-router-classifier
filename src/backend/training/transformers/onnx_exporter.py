"""
ONNX export for fine-tuned transformer classifiers.

Exports a HuggingFace checkpoint to ONNX using optimum.onnxruntime.
The exported model is ~3–5× faster at inference vs PyTorch on CPU,
which is critical for the <30ms latency target in production routing.

EXPORTED MODEL SPEC:
    Input:  input_ids (batch, max_length), attention_mask (batch, max_length)
    Output: logits (batch, 2) → softmax → P(label=0), P(label=1)
    max_length: 64 (covers 95th-percentile of Sprout queries)

OUTPUT DIRECTORY:
    experiments/{dataset}/transformers/models/{model_key}_onnx/

Usage:
    exporter = OnnxExporter()
    result   = exporter.export("v1", "xlmr-base")
    print(result.output_dir)

Or from CLI:
    python phases/phase_6_train_transformers.py --export-onnx --model xlmr-base
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from backend.shared.path_resolver import get_experiment_path
from backend.training.transformers.config import TRANSFORMER_REGISTRY

log = logging.getLogger(__name__)


@dataclass
class OnnxExportResult:
    """Result of one ONNX export."""

    model_key: str
    dataset_name: str
    checkpoint_dir: str
    output_dir: str
    max_length: int


class OnnxExporter:
    """Exports a trained HuggingFace model checkpoint to ONNX format."""

    def export(
        self,
        dataset_name: str,
        model_key: str,
        max_length: int = 64,
        quantize: bool = False,
    ) -> OnnxExportResult:
        """Export a fine-tuned checkpoint to ONNX.

        Args:
            dataset_name: Dataset version, e.g. "v1".
            model_key: Key from TRANSFORMER_REGISTRY (e.g. "xlmr-base").
            max_length: Token sequence length. Must match training max_length.
            quantize: If True, apply INT8 dynamic quantization after export.
                      Reduces model size by ~4× with minimal accuracy loss.

        Returns:
            OnnxExportResult with checkpoint_dir and output_dir paths.
        """
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        if model_key not in TRANSFORMER_REGISTRY:
            raise ValueError(
                f"Unknown model key {model_key!r}. Available: {sorted(TRANSFORMER_REGISTRY)}"
            )

        models_dir = get_experiment_path(dataset_name, "transformers") / "models"
        checkpoint_dir = models_dir / model_key
        output_dir = models_dir / f"{model_key}_onnx"

        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_dir}. "
                f"Run TransformerTrainer.train_experiment('{dataset_name}', '{model_key}') first."
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Exporting {model_key} to ONNX → {output_dir}")

        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
        tokenizer.save_pretrained(str(output_dir))

        model = ORTModelForSequenceClassification.from_pretrained(
            str(checkpoint_dir),
            export=True,
        )
        model.save_pretrained(str(output_dir))

        if quantize:
            output_dir = quantize_onnx(output_dir, model_key)

        log.info(f"Export complete: {output_dir}")
        log.info(f"Inference: use OnnxPredictor.from_dir('{output_dir}')")

        return OnnxExportResult(
            model_key=model_key,
            dataset_name=dataset_name,
            checkpoint_dir=str(checkpoint_dir),
            output_dir=str(output_dir),
            max_length=max_length,
        )

    def export_all(
        self,
        dataset_name: str,
        max_length: int = 64,
        quantize: bool = False,
        skip_on_failure: bool = True,
    ) -> list[OnnxExportResult]:
        """Export all trained models in TRANSFORMER_REGISTRY."""
        results: list[OnnxExportResult] = []
        for model_key in TRANSFORMER_REGISTRY:
            try:
                result = self.export(dataset_name, model_key, max_length, quantize)
                results.append(result)
            except FileNotFoundError:
                log.warning(f"Skipping {model_key} — checkpoint not found")
            except Exception as exc:  # noqa: BLE001
                if skip_on_failure:
                    log.error(f"Export failed for {model_key}: {exc}")
                else:
                    raise
        return results


def quantize_onnx(model_dir: Path, model_key: str) -> Path:
    """Apply INT8 dynamic quantization to the exported ONNX model.

    Reduces model size ~4× and speeds up CPU inference.
    Accuracy impact is typically < 0.5% on classification tasks.

    Args:
        model_dir: Directory containing the ONNX model.
        model_key: Used to name the quantized output directory.

    Returns:
        Path to the quantized model directory.
    """
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    quantized_dir = model_dir.parent / f"{model_key}_onnx_quantized"
    quantized_dir.mkdir(parents=True, exist_ok=True)

    quantizer = ORTQuantizer.from_pretrained(str(model_dir))
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

    quantizer.quantize(
        save_dir=str(quantized_dir),
        quantization_config=qconfig,
    )

    log.info(f"Quantized model saved: {quantized_dir}")
    return quantized_dir
