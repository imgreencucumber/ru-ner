from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def export_onnx(model_dir, onnx_path):
    """Export a token classification model to ONNX with dynamic batch and sequence axes.

    Weights end up in a separate `<name>.onnx.data` file next to the graph.
    """
    # eager attention exports to plain MatMul/Softmax ops, which the int8 quantizer handles
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir, attn_implementation="eager"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    sample = tokenizer(
        [["Глава", "Минфина", "Антон", "Силуанов"], ["В", "Москве"]],
        is_split_into_words=True,
        padding=True,
        return_tensors="pt",
    )
    batch, seq = torch.export.Dim("batch"), torch.export.Dim("sequence")
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        args=(),
        kwargs=dict(sample),
        f=str(onnx_path),
        input_names=list(sample),
        output_names=["logits"],
        dynamic_shapes={name: {0: batch, 1: seq} for name in sample},
        dynamo=True,
        opset_version=18,
    )


def optimize_for_cpu(onnx_path, out_path, num_heads=12, hidden_size=768):
    """Fuse BERT subgraphs into single ONNX Runtime ops (SkipLayerNormalization, BiasGelu).

    The attention block of a dynamo export doesn't match ORT's fusion patterns and stays unfused.
    """
    from onnxruntime.transformers import optimizer

    model = optimizer.optimize_model(
        str(onnx_path), model_type="bert", num_heads=num_heads, hidden_size=hidden_size
    )
    model.save_model_to_file(str(out_path), use_external_data_format=True)
    return model.get_fused_operator_statistics()


def quantize_int8(onnx_path, int8_path):
    """Dynamic quantization: weights are stored as int8, activations are quantized on the fly.

    reduce_range=True uses 7-bit weights. On CPUs without VNNI instructions (like the i3-8100
    used here) int8 matrix multiplication can overflow intermediate sums, 7 bits avoid that.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(onnx_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
    )


def max_logit_diff(model_dir, onnx_path, sentences):
    """Largest absolute difference between PyTorch and ONNX Runtime logits on the same input.

    Padding positions are excluded: their logits are never used and can differ arbitrarily.
    """
    import onnxruntime as ort

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    enc = tokenizer(sentences, is_split_into_words=True, padding=True, return_tensors="np")
    with torch.inference_mode():
        ref = model(**{k: torch.from_numpy(v) for k, v in enc.items()}).logits.numpy()
    names = [i.name for i in session.get_inputs()]
    out = session.run(["logits"], {n: enc[n].astype(np.int64) for n in names})[0]
    real_tokens = enc["attention_mask"].astype(bool)
    return float(np.abs(out - ref)[real_tokens].max())


def model_size_mb(onnx_path):
    """Graph file plus external weights file, if any."""
    path = Path(onnx_path)
    files = [path, path.with_name(path.name + ".data")]
    return sum(f.stat().st_size for f in files if f.exists()) / 2**20
