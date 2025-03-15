def check_cnn1d_sizes(
    seq_len: int, n_layers: int, padding: int, kernel_size: int, pool_size: int
) -> tuple[bool, str]:
    """
    Check if kernel_size and pool_size are valid for a given seq_len.
    Returns: (is_valid, message)
    """
    curr_length = seq_len
    for layer in range(n_layers):
        # Check kernel_size for Conv1d
        padded_length = curr_length + 2 * padding
        if padded_length < kernel_size:
            return (
                False,
                f"Layer {layer + 1}: kernel_size {kernel_size} > padded length {padded_length}",
            )

        # Update length after convolution
        curr_length = padded_length - kernel_size + 1
        if curr_length <= 0:
            return (
                False,
                f"Layer {layer + 1}: curr_length {curr_length} <= 0 after Conv1d",
            )

        # Update length after pooling
        curr_length //= pool_size
        if curr_length <= 0:
            return (
                False,
                f"Layer {layer + 1}: curr_length {curr_length} <= 0 after MaxPool1d (pool_size={pool_size})",
            )

    # Final check
    if curr_length <= 0:
        return False, "Final curr_length <= 0, invalid for Linear layer"

    return True, f"Valid: final curr_length = {curr_length}"


class WindowSlider:
    def __init__(self, capacity: int, features, s=None, e=None):
        self.arr = np.empty((0, features))
        self.capacity = capacity
        self.s = s
        self.e = e

    def push(self, new_row):
        self.arr = np.append(self.arr, [new_row], axis=0)
        if len(self.arr) > self.capacity:
            self.arr = self.arr[1:]

    def get(self, norm=True):
        if norm:
            part = self.arr[:, self.s : self.e]
            norm_part = scaler.fit_transform(part)
            self.arr[:, self.s : self.e] = norm_part
            return self.arr
        else:
            return self.arr


def make_mask(mask):
    mask = torch.tensor(mask, dtype=torch.float)
    return mask.masked_fill(mask == 0, float("-1e10"))


class StochasticActor:
    def __init__(self, actor_net, categorical=False):
        self.actor_net = actor_net
        self.categorical = categorical

    def __call__(self, x, mask=None, min=-float("inf"), max=float("inf")):
        if self.categorical:
            logits = self.actor_net(x)

            mask = make_mask(mask or [1] * len(logits))
            logits = logits * mask

            probs = F.softmax(logits, dim=0)
            dist = Categorical(probs)
            action = dist.sample()
        else:
            mean, std = self.actor_net(x)
            std = F.softplus(std)
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, min, max)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate(self, x, actions):
        if self.categorical:
            logits = self.actor_net(x)
            probs = F.softmax(logits, dim=0)
            dist = Categorical(probs)
        else:
            mean, std = self.actor_net(x)
            std = F.softplus(std)
            dist = Normal(mean, std)

        return dist.log_prob(actions)


class RolloutBuffer:
    def __init__(self):
        self.data = defaultdict(list)

    def add(self, entries):
        for key, value in entries.items():
            self.data[key].append(value)

    def __getitem__(self, key):
        if key in self.data:
            return torch.stack(self.data[key])
        return torch.tensor([])

    def sample(self):
        key = next(iter(self.data))
        length = len(self.data[key])

        indices = np.random.permutation(length).tolist()
        return indices

    def clear(self):
        self.data = defaultdict(list)


class GAE:
    def __init__(self, gamma, lmbda, value_net):
        self.gamma = gamma
        self.lmbda = lmbda
        self.value_net = value_net

    def __call__(self, states, next_states, rewards, dones):
        values = self.value_net(states)
        next_values = self.value_net(next_states)

        advantages = torch.zeros_like(rewards, dtype=torch.float)
        returns = torch.zeros_like(rewards, dtype=torch.float)
        gae = 0

        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t] + self.gamma * (1 - dones[t]) * next_values[t] - values[t]
            )
            gae = delta + self.gamma * self.lmbda * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages, returns


class Nil:
    def __repr__(self):
        return "nil"

    def __bool__(self):
        return False


nil = Nil()


def r(num, d=0):
    return round(num) if d == 0 else round(num, d)
