import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_loo_candidate_calibration.py")
SPEC = importlib.util.spec_from_file_location("loo_analysis", MODULE_PATH)
loo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loo)


def occurrence(view, nll=1.0):
    return {
        "source_name": f"task_0.{view}.ex01",
        "source_view": view,
        "beam_score": 1.0,
        "score_aug": loo.np.asarray([nll, nll]),
    }


def test_source_view_ignores_example_order():
    assert loo.source_view("task_0.transpose.rot90.permute123.ex210") == (
        "transpose.rot90.permute123"
    )


def test_choose_calibration_uses_heldout_label_to_reweight_sources():
    correct = ((1,),)
    wrong_a = ((2,),)
    wrong_b = ((3,),)
    groups = {
        correct: [occurrence("good")],
        wrong_a: [occurrence("bad"), occurrence("bad")],
        wrong_b: [occurrence("bad"), occurrence("bad")],
    }
    selected = loo.choose_calibration(groups, {"good", "bad"}, correct, betas=(1.0,))
    assert selected["variant"] == "source"
    assert selected["loo_rank"] <= 2


def test_target_label_is_used_only_after_calibration_choice():
    heldout = ((1,),)
    target = ((7,),)
    wrong = ((8,),)
    calibration_groups = {
        "task_0": {
            heldout: [occurrence("good")],
            ((2,),): [occurrence("bad"), occurrence("bad"), occurrence("bad")],
            ((3,),): [occurrence("bad"), occurrence("bad"), occurrence("bad")],
        }
    }
    target_groups = {
        "task_0": {
            target: [occurrence("good")],
            wrong: [occurrence("bad"), occurrence("bad"), occurrence("bad")],
            ((9,),): [occurrence("bad"), occurrence("bad"), occurrence("bad")],
        }
    }
    result = loo.evaluate(
        calibration_groups=calibration_groups,
        calibration_views={"task_0": {"good", "bad"}},
        target_groups=target_groups,
        target_views={"task_0": {"good", "bad"}},
        heldout_outputs={"task": [[1]]},
        target_solutions={"task": [[[7]]]},
        challenges={"task": {"test": [{"input": [[0]]}]}},
        task_keys=["task"],
        betas=(1.0,),
    )
    row = result["rows"][0]
    assert row["selected_variant"] == "source"
    assert not row["baseline_top2_hit"]
    assert row["calibrated_top2_hit"]
    assert result["calibrated_top2"] == 1.0
