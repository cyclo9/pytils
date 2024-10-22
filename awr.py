import torch


def compute_gae(rewards, values, dones, gamma=0.99, lmbda=0.95, norm=True):
    advantages = []
    for r, v, d in zip(rewards, values, dones):
        ep_advantages = torch.zeros_like(r)
        gae = 0

        for t in reversed(range(len(r))):
            if t + 1 < len(r):
                delta = r[t] + gamma * v[t + 1] * (1 - d[t + 1]) - v[t]
            else:
                delta = r[t] - v[t]

            gae = delta + gamma * lmbda * gae
            ep_advantages[t] = gae

        if norm:
            ep_advantages = (ep_advantages - ep_advantages.mean()) / (
                ep_advantages.std() + 1e-10
            )
        advantages.append(ep_advantages)

    return torch.cat(advantages).flatten()


# def compute_gae(
#     batch_rewards, batch_vals, batch_dones, gamma=0.99, lmbda=0.95, norm=True
# ):
#     batch_advantages = []
#
#     for rewards, values, dones in zip(batch_rewards, batch_vals, batch_dones):
#         dones = torch.tensor(dones, dtype=torch.int)
#         deltas = (
#             rewards
#             + gamma
#             * torch.cat([values[1:], torch.zeros(1)])
#             * (1 - torch.cat([dones[1:], torch.zeros(1)]))
#             - values
#         )
#         adv_factors = (gamma * lmbda * (1 - dones)).flip(dims=[0])
#         advantages = torch.flip(
#             torch.cumsum(torch.flip(deltas * adv_factors, dims=[0]), dim=0), dims=[0]
#         )
#
#         batch_advantages.append(advantages)
#
#     flat_advantages = torch.cat(batch_advantages)
#
#     if norm:
#         mean_advantages = flat_advantages.mean()
#         std_advantages = flat_advantages.std()
#         flat_advantages = (flat_advantages - mean_advantages) / (std_advantages + 1e-8)
#
#     return flat_advantages


# def compute_gae(
#     batch_rewards, batch_vals, batch_dones, gamma=0.99, lmbda=0.95, norm=True
# ):
#     batch_advantages = []
#
#     for rewards, values, dones in zip(batch_rewards, batch_vals, batch_dones):
#         T = len(rewards)
#         advantages = torch.zeros(T)
#         gae = 0
#
#         for t in reversed(range(T)):
#             if t + 1 < T:
#                 delta = (
#                     rewards[t] + gamma * values[t + 1] * (1 - dones[t + 1]) - values[t]
#                 )
#             else:
#                 delta = rewards[t] - values[t]
#
#             gae = delta + gamma * lmbda * (1 - dones[t]) * gae
#             advantages[t] = gae
#
#         batch_advantages.append(advantages)
#
#     flat_advantages = torch.cat(batch_advantages)
#
#     return flat_advantages
