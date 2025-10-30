import torch
import numpy as np
from torch.optim.adamw import AdamW
import torch.nn as nn
import torch.nn.functional as F

from loss import clipped_ppo_loss
from actor import StochasticActor, ActorCritic
from awr import compute_gae


def compute_kl(log_p, log_q):
    with torch.no_grad():
        kl = (torch.exp(log_p) * (log_p - log_q)).sum()
    return kl


class PPO:
    def __init__(self, env, n_obs, n_act, actor_net, critic_net, discrete=True):
        self._init_hyperparameters()

        # environment information
        self.env = env
        self.n_obs = n_obs
        self.n_act = n_act

        # initalize actor and critic
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.ac = ActorCritic(actor_net, critic_net, discrete)

        self.actor_optim = AdamW(self.actor_net.parameters(), lr=self.lr, amsgrad=True)
        self.critic_optim = AdamW(
            self.critic_net.parameters(), lr=self.lr, amsgrad=True
        )

        # create covariance matrix; arbitrary std = 0.5
        cov_var = torch.full(size=(self.n_act,), fill_value=0.5)
        self.cov_mat = torch.diag(cov_var)

        self.steps = 0

    def _init_hyperparameters(
        self,
        epochs=10,
        minibatch_size=4,
        gamma=0.99,
        lmbda=0.95,
        clip=0.2,
        lr=1e-4,
        c1=0.01,
    ):
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.clip = clip
        self.lr = lr
        self.c1 = c1

    def learn(
        self,
        batch_obs,
        batch_act,
        batch_log_probs,
        batch_rewards,
        batch_vals,
        batch_dones,
    ):
        # frac = (t - 1.0) / timestamps
        # new_lr = max(self.lr * (1.0 - frac), 0.0)
        # self.actor_optim.param_groups[0]["lr"] = new_lr
        # self.critic_optim.param_groups[0]["lr"] = new_lr

        advantages = compute_gae(batch_rewards, batch_vals, batch_dones, False)
        values, _, _ = self.ac.evaluate(batch_obs, batch_act)
        returns = advantages + values.detach()
        advantages = compute_gae(batch_rewards, batch_vals, batch_dones)

        actor_losses = []
        critic_losses = []

        _, P, _ = self.ac.evaluate(batch_obs, batch_act)

        for _ in range(self.epochs):
            inds = np.arange(len(batch_obs))

            np.random.shuffle(inds)
            for i in range(
                0, len(batch_obs) - self.minibatch_size, self.minibatch_size
            ):
                idx = inds[i : i + self.minibatch_size]
                obs_mini = batch_obs[idx]
                act_mini = batch_act[idx]
                old_probs = batch_log_probs[idx]
                mini_advantages = advantages[idx]
                mini_returns = returns[idx]

                values, curr_probs, entropy = self.ac.evaluate(obs_mini, act_mini)

                actor_loss, critic_loss = clipped_ppo_loss(
                    curr_probs,
                    old_probs,
                    mini_advantages,
                    values,
                    mini_returns,
                    entropy,
                )

                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), 100)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), 100)
                self.critic_optim.step()

        _, Q, _ = self.ac.evaluate(batch_obs, batch_act)

        kl = compute_kl(P, Q)
        avg_actor_loss = sum(actor_losses) / len(actor_losses)
        avg_critic_loss = sum(critic_losses) / len(critic_losses)
        print(f"Actor Loss: {avg_actor_loss:.2f}")
        print(f"Critic Loss: {avg_critic_loss:.2f}")
        print(f"KL: {kl:.2f}")
        print()
