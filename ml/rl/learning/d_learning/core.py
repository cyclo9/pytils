def bandit_q_loss(model, obs, action_idx, reward):
    q_value = model(obs).flatten()[action_idx]
    return 0.5 * (reward - q_value) ** 2
