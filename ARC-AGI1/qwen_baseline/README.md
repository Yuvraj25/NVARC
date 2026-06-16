Frozen Python baseline for the public Qwen ARC inference path.

This package is extracted from `002_ivan_arc1.ipynb`, which the repo README
states mirrors the Kaggle Qwen notebook used in the winning solution, except
for targeting ARC-AGI 2024 public evaluation.

What is frozen here:

- `arc_loader.py`: dataset formatting and ARC augmentations
- `arc_decoder.py`: candidate aggregation and selection
- `arc_search.py`: baseline token-by-token DFS
- `arc_rescoring.py`: baseline likelihood scoring plus prefix-cached rescoring
- `arc_solver.py`: TTFT worker loop that ties everything together
- `starter.py`: multiprocessing entrypoint

The intended workflow on this branch is:

1. Keep this extracted baseline runnable and close to notebook behavior.
2. Make search-side changes only inside `arc_search.py` and `arc_rescoring.py`.
3. Avoid editing the original notebook until the Python path has stabilized.
