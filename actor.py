import torch
from torch.distributions import MultivariateNormal, Categorical
import torch.nn.functional as F


def make_mask(mask):
    mask = torch.tensor(mask, dtype=torch.float)
    return mask.masked_fill(mask == 0, float("-1e10"))


class StochasticActor:
    def __init__(self, actor_net, action_dims, discrete=True):
        self.actor_net = actor_net
        self.action_dims = action_dims
        self.discrete = discrete

    def __call__(self, x, mask=None, min=-float("inf"), max=float("inf")):
        if self.discrete:
            logits = self.actor_net(x)
            mask = make_mask(mask or [1] * len(logits))
            logits = logits * mask

            probs = F.softmax(logits, dim=0)
            dist = Categorical(probs)
            action = dist.sample()
        else:
            # NOTE: assumes an output size of 2 * n
            mean, log_cov = self.actor_net(x)
            cov_matrix = torch.diag_embed(F.softplus(log_cov))

            dist = MultivariateNormal(mean, cov_matrix)
            action = dist.sample()
            action = torch.clamp(action, min, max)

        log_prob = dist.log_prob(action)
        return action.detach(), log_prob.detach()

    def evaluate(self, states, actions, mask=None):
        if self.discrete:
            logits = self.actor_net(states)

            if mask is not None:
                logits = logits * mask

            probs = F.softmax(logits, dim=0)
            dist = Categorical(probs)
        else:
            mean, log_cov = self.actor_net(states)
            cov_matrix = torch.diag_embed(F.softplus(log_cov))

            dist = MultivariateNormal(mean, cov_matrix)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy
