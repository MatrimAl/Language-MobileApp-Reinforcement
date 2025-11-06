import torch

from rl import AgentRegistry


STATE_DIM = 13
N_ACTIONS = 5


def test_agent_registry_uses_snapshot_for_new_users(tmp_path):
	snapshot_path = tmp_path / "global_agent.pt"
	registry = AgentRegistry(
		state_dim=STATE_DIM,
		n_actions=N_ACTIONS,
		snapshot_path=snapshot_path,
	)

	agent = registry.get(1)
	for param in agent.q.parameters():
		torch.nn.init.constant_(param, 0.5)
	for param in agent.tgt.parameters():
		torch.nn.init.constant_(param, 0.25)

	registry.global_state = agent.dump_weights()
	registry._save_snapshot(user_id=1)

	# Yeni registry snapshot'tan yüklemeli
	registry2 = AgentRegistry(
		state_dim=STATE_DIM,
		n_actions=N_ACTIONS,
		snapshot_path=snapshot_path,
	)
	agent2 = registry2.get(2)

	saved_state = torch.load(snapshot_path)["state_dict"]

	for name, tensor in agent2.q.state_dict().items():
		assert torch.allclose(tensor, saved_state["q"][name])
	for name, tensor in agent2.tgt.state_dict().items():
		assert torch.allclose(tensor, saved_state["tgt"][name])


def test_record_training_promotes_global_state(tmp_path):
	snapshot_path = tmp_path / "global_agent.pt"
	registry = AgentRegistry(
		state_dim=STATE_DIM,
		n_actions=N_ACTIONS,
		snapshot_path=snapshot_path,
		promote_every=1,
		min_buffer_for_promo=4,
		soft_update_tau=1.0,
	)

	agent = registry.get(1)
	dummy_state = [0.0] * STATE_DIM
	for _ in range(4):
		agent.push(dummy_state, 0, 0.0, dummy_state, False)

	registry.record_training(user_id=1, agent=agent, loss=0.5)

	assert registry.global_state is not None
	assert snapshot_path.exists()
