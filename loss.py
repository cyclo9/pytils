import torch
import torch.nn.functional as F


def clipped_ppo_loss(
    curr_probs,
    old_probs,
    advantages,
    values,
    returns,
    entropy,
    clip=0.2,
    c1=0.01,
):
    ratio = torch.exp(curr_probs - old_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages

    actor_loss = (-torch.min(surr1, surr2)).mean()
    actor_loss = actor_loss - (c1 * entropy.mean())

    critic_loss = F.smooth_l1_loss(values, returns)

    return actor_loss, critic_loss
