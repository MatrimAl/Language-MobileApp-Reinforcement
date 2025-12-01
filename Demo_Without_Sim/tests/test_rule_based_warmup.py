from types import SimpleNamespace

from app import cold_start_action, compute_dynamic_eps
from model import LEVELS


def _build_state(target_level: str, target_acc: float = 0.6) -> list[float]:
	target_idx = LEVELS.index(target_level)
	accs = [0.5] * len(LEVELS)
	accs[target_idx] = target_acc
	extra = [0.6, 0.4, 0.3]
	target_one_hot = [1.0 if i == target_idx else 0.0 for i in range(len(LEVELS))]
	return accs + extra + target_one_hot


def test_cold_start_returns_target_level_initially(monkeypatch):
	user = SimpleNamespace(target_level="A2")
	state = _build_state("A2", target_acc=0.4)
	monkeypatch.setattr("random.random", lambda: 0.5)

	action = cold_start_action(user, state, attempt_count=0)

	assert action == LEVELS.index("A2")


def test_cold_start_allows_step_down_when_struggling(monkeypatch):
	user = SimpleNamespace(target_level="A2")
	state = _build_state("A2", target_acc=0.3)
	monkeypatch.setattr("random.random", lambda: 0.5)

	action = cold_start_action(user, state, attempt_count=12)

	assert action == LEVELS.index("A1")


def test_cold_start_relaxes_after_threshold(monkeypatch):
	user = SimpleNamespace(target_level="A2")
	state = _build_state("A2", target_acc=0.7)
	monkeypatch.setattr("random.random", lambda: 0.5)

	action = cold_start_action(user, state, attempt_count=40)

	assert action is None


def test_dynamic_epsilon_schedule():
	assert compute_dynamic_eps(0) == 0.35
	assert compute_dynamic_eps(25) == 0.25
	assert compute_dynamic_eps(80) == 0.18
	assert compute_dynamic_eps(200) == 0.12
	assert compute_dynamic_eps(400) == 0.08
