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
# join_human_scores() -- re-attach system + judge score to the blind
# sheet, with exactly-once join validation
# ---------------------------------------------------------------------------

def _make_human_df(n=9):
    return pd.DataFrame({
        "item_id": [f"h{i:02d}" for i in range(n)],
        "message": [f"msg {i}" for i in range(n)],
        "reply": [f"reply {i}" for i in range(n)],
        "reference": [f"ref {i}" for i in range(n)],
        "human_overall": [((i + 1) % 5) + 1 for i in range(n)],
    })


def _make_key_df(n=9):
    systems = ["trivial", "nearest", "grounded"]
    return pd.DataFrame({
        "item_id": [f"h{i:02d}" for i in range(n)],
        "pair_id": range(1, n + 1),
        "root_id": [f"r{i}" for i in range(n)],
        "system": [systems[i % 3] for i in range(n)],
    })


def _make_reply_rows_df(n=9):
    systems = ["trivial", "nearest", "grounded"]
    return pd.DataFrame({
        "root_id": [f"r{i}" for i in range(n)],
        "system": [systems[i % 3] for i in range(n)],
        "overall": [(i % 5) + 1 for i in range(n)],
    })


def test_join_human_scores_reattaches_system_and_judge_score():
    n = 9
    human_df, key_df, reply_rows = _make_human_df(n), _make_key_df(n), _make_reply_rows_df(n)
    df = ha.join_human_scores(human_df, key_df, reply_rows)
    assert set(df.columns) == {"pair_id", "root_id", "message", "system",
                                "reply", "reference", "judge_overall", "human_overall"}
    assert len(df) == n
    assert set(df["system"]) == {"trivial", "nearest", "grounded"}
    # row 0: item_id h00 -> root_id r0 -> reply_rows overall for (r0, trivial)
    row0 = df[df["root_id"] == "r0"].iloc[0]
    assert row0["judge_overall"] == 1  # (0 % 5) + 1


def test_join_human_scores_shuffled_order_still_joins_correctly():
    # The blind sheet is shuffled relative to the key/reply_rows -- the join
    # must be by item_id, not by row position.
    human_df = _make_human_df(3).iloc[[2, 0, 1]].reset_index(drop=True)
    key_df, reply_rows = _make_key_df(3), _make_reply_rows_df(3)
    df = ha.join_human_scores(human_df, key_df, reply_rows)
    row = df[df["message"] == "msg 2"].iloc[0]
    assert row["root_id"] == "r2"


def test_join_human_scores_raises_on_missing_human_column():
    human_df = _make_human_df(3).drop(columns=["reference"])
    with pytest.raises(ValueError, match="reference"):
        ha.join_human_scores(human_df, _make_key_df(3), _make_reply_rows_df(3))


def test_join_human_scores_raises_on_item_id_not_in_key():
    human_df = _make_human_df(3)
    human_df.loc[0, "item_id"] = "h99"
    with pytest.raises(ValueError, match="h99"):
        ha.join_human_scores(human_df, _make_key_df(3), _make_reply_rows_df(3))


def test_join_human_scores_raises_on_key_item_missing_from_human_scores():
    key_df = _make_key_df(3)
    key_df.loc[3] = {"item_id": "h99", "pair_id": 4, "root_id": "r99", "system": "trivial"}
    with pytest.raises(ValueError, match="h99"):
        ha.join_human_scores(_make_human_df(3), key_df, _make_reply_rows_df(3))


def test_join_human_scores_raises_on_duplicate_item_id_in_human_scores():
    human_df = _make_human_df(3)
    human_df.loc[2, "item_id"] = human_df.loc[0, "item_id"]  # duplicate h00
    with pytest.raises(ValueError, match="duplicate item_id"):
        ha.join_human_scores(human_df, _make_key_df(3), _make_reply_rows_df(3))


def test_join_human_scores_raises_on_missing_reply_rows_entry():
    reply_rows = _make_reply_rows_df(3)
    reply_rows = reply_rows[reply_rows["root_id"] != "r0"]  # drop r0's judge score
    with pytest.raises(ValueError, match="r0"):
        ha.join_human_scores(_make_human_df(3), _make_key_df(3), reply_rows)


def test_join_human_scores_raises_on_duplicate_reply_rows_key():
    reply_rows = _make_reply_rows_df(3)
    dup = reply_rows.iloc[[0]].copy()
    reply_rows = pd.concat([reply_rows, dup], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        ha.join_human_scores(_make_human_df(3), _make_key_df(3), reply_rows)


# ---------------------------------------------------------------------------
# main() -- file I/O, validation, per-system grouping, output
# ---------------------------------------------------------------------------

def _write_blind_inputs(tmp_path, n=9, human_overall=None):
    golden_dir = tmp_path / "golden"
    results_dir = tmp_path / "results"
    golden_dir.mkdir()
    results_dir.mkdir()

    human_df = _make_human_df(n)
    if human_overall is not None:
        human_df["human_overall"] = human_overall
    human_df.to_csv(golden_dir / "human_scores.csv", index=False)
    _make_key_df(n).to_csv(results_dir / "human_scoring_key.csv", index=False)
    _make_reply_rows_df(n).to_csv(results_dir / "reply_rows.csv", index=False)
    return golden_dir, results_dir


def test_main_missing_file_returns_none(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ha.config, "GOLDEN_DIR", tmp_path)
    monkeypatch.setattr(ha.config, "RESULTS_DIR", tmp_path / "results")
    result = ha.main()
    assert result is None
    out = capsys.readouterr().out
    assert "human_scores.csv" in out
    assert "run" in out.lower() or "eval" in out.lower()


def test_main_full_run_writes_json_with_per_system_and_confusion(tmp_path, monkeypatch):
    golden_dir, results_dir = _write_blind_inputs(tmp_path, n=9)

    monkeypatch.setattr(ha.config, "GOLDEN_DIR", golden_dir)
    monkeypatch.setattr(ha.config, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(ha, "N_BOOTSTRAP", 20)  # keep the test fast

    result = ha.main()

    assert result is not None
    assert set(result.keys()) >= {"overall", "per_system", "confusion",
                                   "bootstrap_ci_kappa_binned", "note"}
    assert result["overall"]["n"] == 9
    assert set(result["per_system"].keys()) == {"trivial", "nearest", "grounded"}
    for sys_metrics in result["per_system"].values():
        assert sys_metrics["n"] == 3
    assert "per-system" in result["note"].lower() or "per_system" in result["note"].lower()

    out_path = results_dir / "judge_human_agreement.json"
    assert out_path.exists()
    saved = json.loads(out_path.read_text())
    assert saved["overall"]["n"] == 9


def test_main_raises_clear_error_on_blank_human_score(tmp_path, monkeypatch):
    scores = [((i + 1) % 5) + 1 for i in range(5)]
    scores = [float(s) for s in scores]
    scores[2] = float("nan")
    golden_dir, results_dir = _write_blind_inputs(tmp_path, n=5, human_overall=scores)

    monkeypatch.setattr(ha.config, "GOLDEN_DIR", golden_dir)
    monkeypatch.setattr(ha.config, "RESULTS_DIR", results_dir)

    with pytest.raises(ValueError, match=r"1 row"):
        ha.main()


def test_main_raises_on_out_of_range_score(tmp_path, monkeypatch):
    scores = [((i + 1) % 5) + 1 for i in range(5)]
    scores[0] = 7
    golden_dir, results_dir = _write_blind_inputs(tmp_path, n=5, human_overall=scores)

    monkeypatch.setattr(ha.config, "GOLDEN_DIR", golden_dir)
    monkeypatch.setattr(ha.config, "RESULTS_DIR", results_dir)

    with pytest.raises(ValueError, match=r"1\.\.5|1-5|range"):
        ha.main()


def test_main_raises_on_non_integer_human_score(tmp_path, monkeypatch):
    scores = [float(((i + 1) % 5) + 1) for i in range(5)]
    scores[1] = 3.5
    golden_dir, results_dir = _write_blind_inputs(tmp_path, n=5, human_overall=scores)

    monkeypatch.setattr(ha.config, "GOLDEN_DIR", golden_dir)
    monkeypatch.setattr(ha.config, "RESULTS_DIR", results_dir)

    with pytest.raises(ValueError, match=r"integer"):
        ha.main()


def test_main_accepts_integral_float_human_score(tmp_path, monkeypatch):
    # A column that had a blank elsewhere in the raw CSV gets read back by
    # pandas as float64 even for the fully-populated rows (e.g. 4.0 instead
    # of 4) -- that's still a valid integer score and must be accepted.
    scores = [float(((i + 1) % 5) + 1) for i in range(5)]
    golden_dir, results_dir = _write_blind_inputs(tmp_path, n=5, human_overall=scores)

    monkeypatch.setattr(ha.config, "GOLDEN_DIR", golden_dir)
    monkeypatch.setattr(ha.config, "RESULTS_DIR", results_dir)

    result = ha.main()
    assert result is not None
    assert result["overall"]["n"] == 5


def test_written_json_has_no_bare_nan_for_degenerate_group(tmp_path, monkeypatch):
    # A system whose scores are all identical produces NaN kappa/spearman for
    # that per-system group (plausible with ~13 rows/system in the real
    # data). The written JSON must be strict-parser-safe: no bare NaN token.
    n = 9
    golden_dir = tmp_path / "golden"
    results_dir = tmp_path / "results"
    golden_dir.mkdir()
    results_dir.mkdir()

    # first 3 items all belong to "trivial" (not the round-robin rotation
    # _make_key_df uses), so their degenerate all-3 scores actually land in
    # one system group instead of being spread across all three.
    systems = ["trivial"] * 3 + ["nearest"] * 3 + ["grounded"] * 3

    human_df = _make_human_df(n)
    human_df["human_overall"] = [3, 3, 3] + [((i + 1) % 5) + 1 for i in range(6)]
    human_df.to_csv(golden_dir / "human_scores.csv", index=False)

    key_df = _make_key_df(n)
    key_df["system"] = systems
    key_df.to_csv(results_dir / "human_scoring_key.csv", index=False)

    reply_rows = _make_reply_rows_df(n)
    reply_rows["system"] = systems
    reply_rows["overall"] = [3, 3, 3] + [(i % 5) + 1 for i in range(6)]
    reply_rows.to_csv(results_dir / "reply_rows.csv", index=False)

    monkeypatch.setattr(ha.config, "GOLDEN_DIR", golden_dir)
    monkeypatch.setattr(ha.config, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(ha, "N_BOOTSTRAP", 20)

    result = ha.main()
    # sanity: the degenerate group really does produce NaN in-memory
    trivial_kappa = result["per_system"]["trivial"]["cohen_kappa_binned"]
    assert trivial_kappa != trivial_kappa  # NaN

    text = (results_dir / "judge_human_agreement.json").read_text()

    def _reject_constant(token):
        raise ValueError(f"strict JSON parser rejects bare constant: {token}")

    parsed = json.loads(text, parse_constant=_reject_constant)
    assert parsed["per_system"]["trivial"]["cohen_kappa_binned"] is None


def test_main_raises_on_missing_column(tmp_path, monkeypatch):
    golden_dir = tmp_path / "golden"
    results_dir = tmp_path / "results"
    golden_dir.mkdir()
    results_dir.mkdir()
    _make_human_df(5).drop(columns=["reference"]).to_csv(golden_dir / "human_scores.csv", index=False)
    _make_key_df(5).to_csv(results_dir / "human_scoring_key.csv", index=False)
    _make_reply_rows_df(5).to_csv(results_dir / "reply_rows.csv", index=False)

    monkeypatch.setattr(ha.config, "GOLDEN_DIR", golden_dir)
    monkeypatch.setattr(ha.config, "RESULTS_DIR", results_dir)

    with pytest.raises(ValueError, match="reference"):
        ha.main()
