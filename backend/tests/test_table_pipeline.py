from autoflow_scheduling.knowledge.table_pipeline import _quality


def test_well_formed_table_scores_good() -> None:
    rows = [
        ["Engine", "Power", "Torque"],
        ["1.0 TSI", "70 kW", "160 Nm"],
        ["1.4 TSI", "103 kW", "250 Nm"],
    ]

    score, status, consistency, empty, short, coverage, watermark, reason = _quality(
        rows,
        [(0, 0, 1, 1, "Engine"), (0, 0, 1, 1, "Power"), (0, 0, 1, 1, "Torque")],
        set(),
    )

    assert score >= 0.8
    assert status == "good"
    assert consistency == 1
    assert empty == 0
    assert coverage == 1
    assert watermark == 0
    assert reason is None


def test_watermark_and_missing_cells_lower_table_quality() -> None:
    rows = [
        ["Engine", "Power", "Torque"],
        ["1.0", "", "160"],
        ["1.4", "103", ""],
    ]

    score, status, *_rest = _quality(
        rows,
        [
            (0, 0, 1, 1, "Engine"),
            (0, 0, 1, 1, "Protected"),
            (0, 0, 1, 1, "Power"),
        ],
        {"protected"},
    )

    assert score < 0.8
    assert status in {"warning", "failed"}
