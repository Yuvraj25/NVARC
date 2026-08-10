#!/usr/bin/env python3
"""Patch pinned Unsloth 2025.9.7 for cached Qwen3 q_len > 1 inference.

This patch must run before importing ``unsloth``. It deliberately targets the
winner-patched Qwen3 source and fails if the expected source is not present.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "ARC_QWEN3_MULTITOKEN_CACHE_PATCH_V1"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one source match, found {count}")
    return text.replace(old, new, 1)


def patch_llama_source(text: str) -> str:
    if PATCH_MARKER in text:
        return text

    text = _replace_once(
        text,
        """        assert(q_len == 1)\n        # Get saved buffers to reduce memory movement\n""",
        f"""        if q_len < 1:\n            raise ValueError(\"Cached inference requires at least one token\")\n        # {PATCH_MARKER}\n        # Get saved buffers to reduce memory movement\n""",
        "llama cached-inference q_len guard",
    )
    text = _replace_once(
        text,
        """        temp_mlp = torch.empty((2, bsz, 1, mlp_size), dtype = X.dtype, device = f\"{DEVICE_TYPE}:0\")\n""",
        """        temp_mlp = torch.empty((2, bsz, q_len, mlp_size), dtype = X.dtype, device = f\"{DEVICE_TYPE}:0\")\n""",
        "llama MLP scratch shape",
    )
    return text


def patch_qwen3_source(text: str) -> str:
    if PATCH_MARKER in text:
        return text
    if "A = flash_attn_func(Qnn, Knn, Vnn)" not in text:
        raise RuntimeError(
            "Qwen3 source does not contain the winner FlashAttention patch; "
            "apply pip-install-unsloth-flash-patch.ipynb first"
        )

    replacements = [
        (
            "    bsz, _, hd = hidden_states.size()\n",
            f"    bsz, q_len, hd = hidden_states.size()\n    # {PATCH_MARKER}\n",
            "Qwen3 q_len capture",
        ),
        (
            "    kv_seq_len = seq_len + 1\n",
            "    kv_seq_len = seq_len + q_len\n",
            "Qwen3 KV length",
        ),
        (
            "KV_CACHE_INCREMENT+seq_len+1, 2, bsz",
            "KV_CACHE_INCREMENT+seq_len+q_len, 2, bsz",
            "Qwen3 initial paged-cache capacity",
        ),
        (
            "self.temp_QA = torch.empty((2, bsz, 1, attention_size)",
            "self.temp_QA = torch.empty((2, bsz, q_len, attention_size)",
            "Qwen3 QA scratch shape",
        ),
        (
            "self.temp_KV = torch.empty((2, bsz, 1, n_kv_heads*head_dim)",
            "self.temp_KV = torch.empty((2, bsz, q_len, n_kv_heads*head_dim)",
            "Qwen3 KV scratch shape",
        ),
        (
            "self.RH_Q = torch.empty((bsz, n_heads, 1, head_dim)",
            "self.RH_Q = torch.empty((bsz, n_heads, q_len, head_dim)",
            "Qwen3 rotary scratch shape",
        ),
        (
            "self.temp_O = torch.empty((bsz, 1, hidden_size)",
            "self.temp_O = torch.empty((bsz, q_len, hidden_size)",
            "Qwen3 output scratch shape",
        ),
        (
            "self.attention = torch.empty((bsz, n_heads, 1, KV_CACHE_INCREMENT+seq_len)",
            "self.attention = torch.empty((bsz, n_heads, q_len, KV_CACHE_INCREMENT+seq_len+q_len)",
            "Qwen3 attention scratch shape",
        ),
        (
            """    elif kv_seq_len >= self.paged_attention.shape[0]:\n        self.paged_attention.resize_((self.paged_attention.shape[0]+KV_CACHE_INCREMENT, 2, bsz, n_kv_heads, head_dim))\n""",
            """    elif kv_seq_len >= self.paged_attention.shape[0]:\n        new_capacity = max(self.paged_attention.shape[0]+KV_CACHE_INCREMENT, kv_seq_len+1)\n        self.paged_attention.resize_((new_capacity, 2, bsz, n_kv_heads, head_dim))\n""",
            "Qwen3 paged-cache growth",
        ),
        (
            "self.attention.resize_((bsz, n_heads, 1, self.attention.shape[-1]+KV_CACHE_INCREMENT))",
            "self.attention.resize_((bsz, n_heads, q_len, new_capacity))",
            "Qwen3 attention scratch growth",
        ),
        (
            """    pass\n\n    Qn = fast_linear_forward(self.q_proj, Xn, out = self.temp_QA[0])\n""",
            """    pass\n\n    # Draft verification alternates q_len=1 and q_len=K on the same cache.\n    if self.temp_QA.shape[2] != q_len:\n        if attention_size == hidden_size:\n            del self.temp_O\n        self.temp_QA.resize_((2, bsz, q_len, attention_size))\n        self.temp_KV.resize_((2, bsz, q_len, n_kv_heads*head_dim))\n        self.RH_Q.resize_((bsz, n_heads, q_len, head_dim))\n        if attention_size != hidden_size:\n            self.temp_O.resize_((bsz, q_len, hidden_size))\n        else:\n            self.temp_O = self.temp_QA[1][:,:,:hidden_size]\n        self.attention.resize_((bsz, n_heads, q_len, self.paged_attention.shape[0]))\n    pass\n\n    Qn = fast_linear_forward(self.q_proj, Xn, out = self.temp_QA[0])\n""",
            "Qwen3 variable-q_len scratch resizing",
        ),
        (
            "Qn = Qn.view(bsz, 1, n_heads,    head_dim)",
            "Qn = Qn.view(bsz, q_len, n_heads,    head_dim)",
            "Qwen3 Q view",
        ),
        (
            "Kn = Kn.view(bsz, 1, n_kv_heads, head_dim)",
            "Kn = Kn.view(bsz, q_len, n_kv_heads, head_dim)",
            "Qwen3 K view",
        ),
        (
            "Vn = Vn.view(bsz, 1, n_kv_heads, head_dim).transpose(1, 2)",
            "Vn = Vn.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)",
            "Qwen3 V view",
        ),
        (
            "self.rotary_emb.extend_rope_embedding(Vn, seq_len + 2)",
            "self.rotary_emb.extend_rope_embedding(Vn, kv_seq_len + 1)",
            "Qwen3 rotary-cache extension",
        ),
        (
            "self.paged_attention_K[seq_len] = Kn.permute(2, 0, 1, 3)",
            "self.paged_attention_K[seq_len:kv_seq_len] = Kn.permute(2, 0, 1, 3)",
            "Qwen3 K cache append",
        ),
        (
            "self.paged_attention_V[seq_len] = Vn.permute(2, 0, 1, 3)",
            "self.paged_attention_V[seq_len:kv_seq_len] = Vn.permute(2, 0, 1, 3)",
            "Qwen3 V cache append",
        ),
        (
            "A = flash_attn_func(Qnn, Knn, Vnn)",
            "A = flash_attn_func(Qnn, Knn, Vnn, causal = q_len > 1)",
            "Qwen3 causal FlashAttention",
        ),
        (
            "A = A.reshape(bsz, 1, attention_size)",
            "A = A.reshape(bsz, q_len, attention_size)",
            "Qwen3 attention output shape",
        ),
    ]
    for old, new, label in replacements:
        text = _replace_once(text, old, new, label)
    return text


def patch_unsloth(package_dir: Path) -> list[Path]:
    package_dir = package_dir.resolve()
    targets = [
        (package_dir / "models" / "llama.py", patch_llama_source),
        (package_dir / "models" / "qwen3.py", patch_qwen3_source),
    ]
    patched_sources = []
    for path, patcher in targets:
        if not path.is_file():
            raise FileNotFoundError(path)
        original = path.read_text()
        patched = patcher(original)
        if patched != original:
            patched_sources.append((path, patched))
    for path, patched in patched_sources:
        path.write_text(patched)
    return [path for path, _ in patched_sources]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unsloth-package-dir",
        required=True,
        help="Path to the installed unsloth package directory (the folder containing models/)",
    )
    args = parser.parse_args()
    changed = patch_unsloth(Path(args.unsloth_package_dir))
    if changed:
        print("Patched before import:", *(str(path) for path in changed))
    else:
        print("Patch already applied")


if __name__ == "__main__":
    main()
