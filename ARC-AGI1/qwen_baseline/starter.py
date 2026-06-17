import argparse
import json
import os
import time

import torch
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.set_default_device("cpu")

    if rank > 0:
        while not os.path.exists(f"../worker{rank - 1}"):
            time.sleep(5)

    from arc_solver import worker

    with open(f"../worker{rank}", "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start!")
    worker(rank, queue, end_time)
    print(f"[Rank {rank}] done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--use-prefix-cached-rescoring", action="store_true")
    parser.add_argument("--use-speculative-dfs", action="store_true")
    parser.add_argument("--dfs-prob-threshold", type=float, default=0.2)
    args = parser.parse_args()
    end_time = args.end_time if args.end_time is not None else time.time() + 12 * 3600
    os.environ["ARC_USE_PREFIX_CACHED_RESCORING"] = "1" if args.use_prefix_cached_rescoring else "0"
    os.environ["ARC_USE_SPECULATIVE_DFS"] = "1" if args.use_speculative_dfs else "0"
    os.environ["ARC_DFS_PROB_THRESHOLD"] = str(args.dfs_prob_threshold)
    print(
        "runtime flags:",
        f"prefix_cached_rescoring={os.environ['ARC_USE_PREFIX_CACHED_RESCORING']}",
        f"speculative_dfs={os.environ['ARC_USE_SPECULATIVE_DFS']}",
        f"dfs_prob_threshold={os.environ['ARC_DFS_PROB_THRESHOLD']}",
    )

    rerun_mode = True
    if rerun_mode:
        test_path = "../input/arc-prize-2024/arc-agi_evaluation_challenges.json"
    else:
        test_path = "../input/arc-prize-2024/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    queue = mp.Manager().Queue()
    for key in sorted(data.keys()):
        if not rerun_mode and key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:
            continue
        queue.put(key)
    for _ in range(4):
        queue.put(None)

    mp.spawn(local_worker, args=(queue, end_time), nprocs=4)
