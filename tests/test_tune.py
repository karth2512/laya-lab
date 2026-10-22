from scripts.tune import draw_trials


def test_small_space_uses_full_grid() -> None:
    trials = draw_trials({"a": [1, 2], "b": [3]}, n=6)
    assert trials == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def test_large_space_samples_budget_deterministically() -> None:
    space = {"r": [4, 8, 16, 32], "lr": [1, 2, 3], "alpha": [1, 2]}
    trials = draw_trials(space, n=6)
    assert len(trials) == 6 and len({str(t) for t in trials}) == 6
    assert trials == draw_trials(space, n=6)
