from pathlib import Path
import shutil
import tempfile
import unittest

from patch_unsloth_qwen3_multitoken import PATCH_MARKER, patch_unsloth


REPO_ROOT = Path(__file__).resolve().parents[3]
EXTRACTED_UNSLOTH = REPO_ROOT / "extracted" / "unsloth_2025_9_7" / "unsloth" / "unsloth"


def _copy_sources(tmp_path):
    package_dir = tmp_path / "unsloth"
    models_dir = package_dir / "models"
    models_dir.mkdir(parents=True)
    for name in ("llama.py", "qwen3.py"):
        shutil.copy2(EXTRACTED_UNSLOTH / "models" / name, models_dir / name)
    return package_dir


class PatchTest(unittest.TestCase):
    def test_patch_updates_both_sources_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = _copy_sources(Path(temp_dir))
            changed = patch_unsloth(package_dir)
            self.assertEqual({path.name for path in changed}, {"llama.py", "qwen3.py"})

            llama = (package_dir / "models" / "llama.py").read_text()
            qwen3 = (package_dir / "models" / "qwen3.py").read_text()
            self.assertIn(PATCH_MARKER, llama)
            self.assertIn(PATCH_MARKER, qwen3)
            self.assertNotIn("assert(q_len == 1)", llama)
            self.assertIn("(2, bsz, q_len, mlp_size)", llama)
            self.assertIn("kv_seq_len = seq_len + q_len", qwen3)
            self.assertIn("[seq_len:kv_seq_len] = Kn.permute", qwen3)
            self.assertIn("causal = q_len > 1", qwen3)
            self.assertIn("A = A.reshape(bsz, q_len, attention_size)", qwen3)
            self.assertIn("if self.temp_QA.shape[2] != q_len", qwen3)
            self.assertEqual(patch_unsloth(package_dir), [])

    def test_patch_rejects_qwen3_without_winner_flash_patch_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = _copy_sources(Path(temp_dir))
            llama_path = package_dir / "models" / "llama.py"
            original_llama = llama_path.read_text()
            qwen_path = package_dir / "models" / "qwen3.py"
            qwen_path.write_text(
                qwen_path.read_text().replace(
                    "A = flash_attn_func(Qnn, Knn, Vnn)",
                    "A = unsupported_attention(Qnn)",
                )
            )

            with self.assertRaisesRegex(RuntimeError, "winner FlashAttention patch"):
                patch_unsloth(package_dir)
            self.assertEqual(llama_path.read_text(), original_llama)


if __name__ == "__main__":
    unittest.main()
