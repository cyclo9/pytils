import torch
import numpy as np
from torch.optim.adamw import AdamW
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from models import FeedForward
from actor import StochasticActor, ActorCritic


class PPO:
    def __init__(self, env, n_obs, n_act, ac, actor_net, critic_net):
        self._init_hyperparameters()

        # environment information
        self.env = env
        self.n_obs = n_obs
        self.n_act = n_act

        # initalize actor and critic
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.ac = ac

        self.actor_optim = AdamW(self.actor_net.parameters(), lr=self.lr, amsgrad=True)
        self.critic_optim = AdamW(
            self.critic_net.parameters(), lr=self.lr, amsgrad=True
        )

        # create covariance matrix; arbitrary std = 0.5
        cov_var = torch.full(size=(self.n_act,), fill_value=0.5)
        self.cov_mat = torch.diag(cov_var)

        self.steps = 0

    def _init_hyperparameters(self):
        self.steps_per_batch = 200
        self.max_steps_per_episode = 200
        self.gamma = 0.99
        self.epochs = 10
        self.minibatches = 20
        self.lmbda = 0.95
        self.clip = 0.3
        self.lr = 0.0001
        self.c1 = 0.01

    def rollout(self):
        batch_obs = []
        batch_act = []
        batch_log_probs = []
        batch_rewards = []
        batch_returns = []
        batch_lens = []  # episodic length of batch
        batch_vals = []
        batch_dones = []

        ep_rewards = []
        ep_vals = []
        ep_dones = []

        t = 0
        while t < self.steps_per_batch:
            ep_rewards = []  # episodic rewards
            ep_vals = []
            ep_dones = []
            obs, mask = self.env.reset()
            done = False

            for i in range(self.max_steps_per_episode):
                # increment timestaps ran so far
                t += 1
                self.steps += 1
                ep_dones.append(done)

                obs = torch.tensor(obs, dtype=torch.float).unsqueeze(0)
                batch_obs.append(obs)
                action, log_prob = self.ac(obs)
                value = self.critic_net(obs)

                obs, reward, terminated, truncated, mask = self.env.step(action.item())
                done = terminated or truncated

                # collect reward, action, log_prob
                ep_rewards.append(reward)
                ep_vals.append(value)
                batch_act.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # collect episodic length and rewards
            batch_lens.append(i + 1)
            batch_rewards.append(ep_rewards)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

        batch_obs = torch.stack(batch_obs)
        batch_act = torch.tensor(batch_act, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_returns = self.compute_returns(batch_rewards)

        return (
            batch_obs,
            batch_act,
            batch_log_probs,
            batch_rewards,
            batch_returns,
            batch_lens,
            batch_vals,
            batch_dones,
        )

    def learn(self, timestamps):
        t = 0
        while t < timestamps:
            (
                batch_obs,
                batch_act,
                batch_log_probs,
                batch_rewards,
                batch_returns,
                batch_lens,
                batch_vals,
                batch_dones,
            ) = self.rollout()

            # frac = (t - 1.0) / timestamps
            # new_lr = max(self.lr * (1.0 - frac), 0.0)
            # self.actor_optim.param_groups[0]["lr"] = new_lr
            # self.critic_optim.param_groups[0]["lr"] = new_lr

            values, _, _ = self.evaluate(batch_obs, batch_act)
            advantages = self.calculate_gae(batch_rewards, batch_vals, batch_dones)

            values = self.critic_net(batch_obs).squeeze()
            batch_returns = advantages + values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

            print(advantages)
            print(batch_returns)
            raise Exception()

            self.actor_losses = []
            self.critic_losses = []
            self.kl = []

            for _ in range(self.epochs):
                inds = np.arange(len(batch_obs))
                minibatch_size = len(batch_obs) // self.minibatches

                np.random.shuffle(inds)
                for start in range(0, len(batch_obs), minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    obs_mini = batch_obs[idx]
                    act_mini = batch_act[idx]
                    old_probs = batch_log_probs[idx]
                    mini_advantages = advantages[idx]
                    mini_returns = batch_returns[idx]

                    values, curr_probs, entropy = self.evaluate(obs_mini, act_mini)

                    # ratio = torch.exp(curr_log_probs - log_prob_mini)
                    # surr1 = ratio * mini_advantages
                    # surr2 = (
                    #     torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
                    #     * mini_advantages
                    # )
                    #
                    # actor_loss = (-torch.min(surr1, surr2)).mean()
                    # critic_loss = F.smooth_l1_loss(values, mini_returns)

                    ratio = torch.exp(curr_probs - old_probs)
                    surr1 = ratio * mini_advantages
                    surr2 = (
                        torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
                        * mini_advantages
                    )

                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    actor_loss = actor_loss - (self.c1 * entropy.mean())

                    critic_loss = F.smooth_l1_loss(values, mini_returns)

                    print(actor_loss)
                    print(critic_loss)
                    raise Exception()

                    entropy_loss = entropy.mean()
                    actor_loss = actor_loss - (self.c1 * entropy_loss)

                    approx_kl = ((ratio - 1) - (curr_probs - old_probs)).mean()

                    self.actor_losses.append(actor_loss)
                    self.critic_losses.append(critic_loss)
                    self.kl.append(approx_kl)

                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
                    self.actor_optim.step()

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
                    self.critic_optim.step()

            avg_actor_loss = sum(self.actor_losses) / len(self.actor_losses)
            avg_critic_loss = sum(self.critic_losses) / len(self.critic_losses)
            avg_kl = sum(self.kl) / len(self.kl)
            print(f"Steps: {self.steps}")
            print(f"Actor Loss: {avg_actor_loss:.2f}")
            print(f"Critic Loss: {avg_critic_loss:.2f}")
            print(f"KL Divergence: {avg_kl:.2f}")
            print()

            t += np.sum(batch_lens)

    def evaluate(self, obs, acts):
        values = self.critic_net(obs).squeeze()
        log_probs, entropy = self.ac.evaluate(obs, acts)
        return values, log_probs, entropy

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = (
                        ep_rews[t]
                        + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1])
                        - ep_vals[t]
                    )
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = (
                    delta + self.gamma * self.lmbda * (1 - ep_dones[t]) * last_advantage
                )
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float)

    def compute_returns(self, rewards):
        returns = []
        discounted_reward = 0

        for ep_reward in reversed(rewards):
            for reward in reversed(ep_reward):
                discounted_reward = reward + discounted_reward * self.gamma
                returns.append(discounted_reward)

        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float)
        return returns


env = gym.make("CartPole-v1", render_mode="human")

actor_net = FeedForward(4, 2, 2, 64)
critic_net = FeedForward(4, 1, 2, 64)
actor = StochasticActor(actor_net, 2, True)
ac = ActorCritic(actor_net, critic_net, True)

ppo = PPO(env, 4, 2, actor, actor_net, critic_net)
ppo.learn(10000)
