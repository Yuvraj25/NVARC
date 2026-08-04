import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_chunked_sglang_pipeline as pipeline
import starter


def _args(tmpdir):
    return SimpleNamespace(
        test_path=str(Path(tmpdir) / "test.json"),
        model_path=str(Path(tmpdir) / "model"),
        output_dir=str(Path(tmpdir) / "outputs"),
        sglang_tp_size=1,
        sglang_adapter_dir=str(Path(tmpdir) / "adapters"),
        sglang_adapter_manifest=str(Path(tmpdir) / "adapters" / "adapter_manifest.json"),
        dfs_prob_threshold=0.1,
        sglang_speculative_repeat_len=9,
        end_time=None,
        sglang_mem_fraction_static=None,
        profile_timings=True,
        use_speculative_dfs=True,
        sglang_dynamic_repeat=False,
        train_nprocs=2,
        infer_workers=2,
    )


class StreamingManifestTests(unittest.TestCase):
    def test_only_existing_ready_selected_adapters_are_queued(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ready = root / "ready"
            ready.mkdir()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "entries": [
                            {"key": "a", "status": "ready", "adapter_path": str(ready)},
                            {"key": "b", "status": "consumed", "adapter_path": str(ready)},
                            {"key": "c", "status": "ready", "adapter_path": str(root / "missing")},
                        ]
                    }
                )
            )
            args = SimpleNamespace(
                sglang_stream_manifest=str(manifest),
                keys_json=json.dumps(["a", "c"]),
                keys_file=None,
            )
            self.assertEqual(
                starter._load_streaming_manifest_jobs(args, queued_keys=set()),
                [{"key": "a", "adapter_path": str(ready)}],
            )
            self.assertEqual(starter._load_streaming_manifest_jobs(args, queued_keys={"a"}), [])


class StarterProcessTests(unittest.TestCase):
    def test_training_and_streaming_inference_use_disjoint_paths_and_devices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(tmpdir)
            producer_done = Path(tmpdir) / "producer.done"
            base_env = {
                "ARC_SGLANG_STACK_PATH": "/mounted/arc_stack",
                "PYTHONPATH": os.pathsep.join(["/trainer/stack", "/mounted/arc_stack"]),
            }
            with patch.dict(os.environ, base_env, clear=True), patch.object(
                pipeline, "_clear_worker_sentinels"
            ), patch.object(pipeline.subprocess, "Popen") as popen:
                pipeline._start_starter(args, ["a"], "train", cuda_device_offset=0)
                train_cmd = popen.call_args.args[0]
                train_env = popen.call_args.kwargs["env"]
                self.assertEqual(train_cmd[train_cmd.index("--cuda-device-offset") + 1], "0")
                self.assertNotIn("/mounted/arc_stack", train_env.get("PYTHONPATH", "").split(os.pathsep))

                pipeline._start_starter(
                    args,
                    ["a"],
                    "stream-infer",
                    cuda_device_offset=2,
                    producer_done_path=producer_done,
                )
                infer_cmd = popen.call_args.args[0]
                infer_env = popen.call_args.kwargs["env"]
                self.assertEqual(infer_cmd[infer_cmd.index("--cuda-device-offset") + 1], "2")
                self.assertIn("--sglang-stream-manifest", infer_cmd)
                self.assertIn("--sglang-consume-adapters", infer_cmd)
                self.assertIn("/mounted/arc_stack", infer_env["PYTHONPATH"].split(os.pathsep))


class AdapterCapacityTests(unittest.TestCase):
    def test_capacity_counts_only_adapter_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "manifest.json").write_text("{}")
            (root / "a").mkdir()
            (root / "b").mkdir()
            self.assertEqual(pipeline._resident_adapter_count(root), 2)


if __name__ == "__main__":
    unittest.main()
