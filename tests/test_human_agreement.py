import json
import warnings

import pandas as pd
import pytest

from eval import human_agreement as ha


# ---------------------------------------------------------------------------
# agreement() -- core metrics
# ---------------------------------------------------------------------------

def test_perfect_agreement():
    m = ha.agreement([1, 3, 5, 4, 2], [1, 3, 5, 4, 2])
    assert abs(m["cohen_kappa_binned"] - 1.0) < 1e-9
    assert m["n"] == 5
    assert m["exact_agreement"] == pytest.approx(1.0)
    assert m["within_one"] == pytest.approx(1.0)
    assert m["mean_diff"] == pytest.approx(0.0)


def test_partial_agreement_in_range():
    m = ha.agreement([1, 2, 3, 4, 5], [2, 2, 3, 5, 5])
    assert -1.0 <= m["cohen_kappa_binned"] <= 1.0
    assert -1.0 <= m["weighted_kappa_quadratic"] <= 1.0
    assert -1.0 <= m["spearman"] <= 1.0


def test_agreement_returns_all_required_keys():
    m = ha.agreement([1, 2, 3, 4, 5], [2, 2, 3, 5, 5])
    expected = {"n", "cohen_kappa_binned", "weighted_kappa_quadratic", "spearman",
                "exact_agreement", "within_one", "mean_human", "mean_judge", "mean_diff"}
    assert set(m.keys()) == expected


def test_weighted_kappa_rewards_off_by_one_more_than_binned_only_would_suggest():
    # Off-by-one everywhere (judge always one point above human, capped at
    # 5): every score is within one point, so within_one is perfect and the
    # raters are clearly correlated, but binning throws away the near-miss
    # credit at bin boundaries (e.g. human=2/judge=3 crosses low->mid).
    human = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    judge = [min(h + 1, 5) for h in human]
    m = ha.agreement(human, judge)
    assert m["within_one"] == pytest.approx(1.0)
    assert m["exact_agreement"] < 1.0
    assert m["weighted_kappa_quadratic"] > 0.0
    assert m["weighted_kappa_quadratic"] > m["cohen_kappa_binned"]


def test_judge_bias_sign_when_judge_always_scores_one_higher():
    human = [1, 2, 3, 4, 5, 3, 2]
    judge = [x + 1 if x < 5 else x for x in human]
    # keep it simple: judge always exactly human + 1 (cap irrelevant here
    # since we choose values that never hit the ceiling)
    human = [1, 2, 3, 1, 2, 3, 4]
    judge = [2, 3, 4, 2, 3, 4, 5]
    m = ha.agreement(human, judge)
    assert m["mean_diff"] == pytest.approx(1.0)
    assert m["mean_judge"] - m["mean_human"] == pytest.approx(1.0)


def test_degenerate_all_identical_scores_returns_nan_without_warnings():
    human = [3, 3, 3, 3, 3]
    judge = [3, 3, 3, 3, 3]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = ha.agreement(human, judge)
    assert m["cohen_kappa_binned"] != m["cohen_kappa_binned"]  # NaN
    assert m["weighted_kappa_quadratic"] != m["weighted_kappa_quadratic"]  # NaN
    assert m["spearman"] != m["spearman"]  # NaN
    # exact_agreement/within_one/means are still well-defined
    assert m["exact_agreement"] == pytest.approx(1.0)
    assert m["mean_diff"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# confusion_table()
# ---------------------------------------------------------------------------

def test_confusion_table_shape_and_counts():
    human = [1, 1, 3, 5, 5]
    judge = [1, 2, 3, 4, 5]
    ct = ha.confusion_table(human, judge)
    assert ct["labels"] == ["low", "mid", "high"]
    assert len(ct["table"]) == 3 and all(len(row) == 3 for row in ct["table"])
    assert sum(sum(row) for row in ct["table"]) == 5
    # both human=1,judge=1 (low,low) and human=1,judge=2 (low,low) -> table[0][0] == 2
    assert ct["table"][0][0] == 2


# ---------------------------------------------------------------------------
# bootstrap_kappa_ci()
# ---------------------------------------------------------------------------

def test_bootstrap_kappa_ci_is_deterministic_for_fixed_seed():
    human = [1, 2, 3, 4, 5, 3, 2, 4, 1, 5]
    judge = [1, 2, 3, 3, 5, 2, 2, 4, 1, 4]
    r1 = ha.bootstrap_kappa_ci(human, judge, n_boot=200, seed=42)
    r2 = ha.bootstrap_kappa_ci(human, judge, n_boot=200, seed=42)
    assert r1 == r2


def test_bootstrap_kappa_ci_reports_skipped_degenerate_resamples():
    human = [3, 3, 3]
    judge = [3, 3, 3]
    r = ha.bootstrap_kappa_ci(human, judge, n_boot=50, seed=42)
    # every resample of an all-3 array is itself all-3 -> kappa always undefined
    assert r["n_skipped"] == 50
    assert r["lo"] != r["lo"]  # NaN


# ---------------------------------------------------------------------------
# main() -- validation, per-system grouping, output
# ---------------------------------------------------------------------------

_COLUMNS = ("pair_id", "root_id", "message", "system", "reply", "reference",
            "judge_overall", "human_overall")


def _make_df(n=9):
    systems = ["trivial", "nearest", "grounded"]
    return pd.DataFrame({
        "pair_id": range(1, n + 1),
        "root_id": [f"r{i}" for i in range(n)],
        "message": [f"msg {i}" for i in range(n)],
        "system": [systems[i % 3] for i in range(n)],
        "reply": [f"reply {i}" for i in range(n)],
        "reference": [f"ref {i}" for i in range(n)],
        "judge_overall": [(i % 5) + 1 for i in range(n)],
        "human_overall": [((i + 1) % 5) + 1 for i in range(n)],
    })


def test_main_missing_file_returns_none(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ha.config, "GOLDEN_DIR", tmp_path)
    result = ha.main()
    assert result is None
    out = capsys.readouterr().out
    assert "human_scores.csv" in out
    assert "run" in out.lower() or "eval" in out.lower()


def test_main_full_run_writes_json_with_per_system_and_confusion(tmp_path, monkeypatch):
    df = _make_df(n=9)
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    df.to_csv(golden_dir / "human_scores.csv", index=False)
    results_dir = tmp_path / "results"

    monkeypatch.setattr(ha.config, "GOLDEN_DIR", golden_dir)
    monkeypatch.setattr(ha.config, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(ha, "N_BOOTSTRAP", 20)  # keep the test fast

    result = ha.main()

    assert result is not None
    assert set(result.keys()) >= {"overall", "per_system", "confusion", "bootstrap_ci_kappa_binned"}
    assert result["overall"]["n"] == 9
    assert set(result["per_system"].keys()) == {"trivial", "nearest", "grounded"}
    for sys_metrics in result["per_system"].values():
        assert sys_metrics["n"] == 3

    out_path = results_dir / "judge_human_agreement.json"
    assert out_path.exists()
    saved = json.loads(out_path.read_text())
    assert saved["overall"]["n"] == 9


def test_main_raises_clear_error_on_blank_human_score(tmp_path, monkeypatch):
    df = _make_df(n=5)
    df["human_overall"] = df["human_overall"].astype(float)
    df.loc[2, "human_overall"] = float("nan")
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    df.to_csv(golden_dir / "human_scores.csv", index=False)

    monkeypatch.setattr(ha.config, "GOLDEN_DIR", golden_dir)

    with pytest.raises(ValueError, match=r"1 row"):
        ha.main()


def test_main_raises_on_out_of_range_score(tmp_path, monkeypatch):
    df = _make_df(n=5)
    df.loc[0, "human_overall"] = 7
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    df.to_csv(golden_dir / "human_scores.csv", index=False)

    monkeypatch.setattr(ha.config, "GOLDEN_DIR", golden_dir)

    with pytest.raises(ValueError, match=r"1\.\.5|1-5|range"):
        ha.main()


def test_main_raises_on_missing_column(tmp_path, monkeypatch):
    df = _make_df(n=5).drop(columns=["reference"])
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    df.to_csv(golden_dir / "human_scores.csv", index=False)

    monkeypatch.setattr(ha.config, "GOLDEN_DIR", golden_dir)

    with pytest.raises(ValueError, match="reference"):
        ha.main()
