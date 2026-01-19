import torch
import torch.nn as nn


class DecisionModel:
    def __init__(self, ensemble, device, state_size, d_model, S):
        self.ensemble = ensemble
        self.R_embed = nn.Linear(1, d_model).to(device)
        self.s_embed = nn.Linear(state_size, d_model).to(device)
        self.a_embed = nn.Linear(1, d_model).to(device)
        self.S = S

    def embed(self, R, s, a):
        R_emb = self.R_embed(R)
        s_emb = self.s_embed(s)
        a_emb = self.a_embed(a)
        return R_emb, s_emb, a_emb

    def _interleave_ids(self):
        pass

    def shape_X_y(self, R, s, a):
        R_emb, s_emb, a_emb = self.embed(R, s, a)

        assert self.S <= R.shape[0], f"S={self.S} exceeds tensor length {R.shape[0]}"
        flat_ids = torch.arange(R.shape[0])
        n = (flat_ids.shape[0] // self.S) * self.S
        ids = flat_ids[:n].view(-1, self.S)
        full_ids = ids
        true_action_ids = ids[:, -1:].squeeze(-1)

        full_cycles = [R_emb[full_ids], s_emb[full_ids], a_emb[full_ids]]
        b, s, f = full_cycles[0].shape
        stacked = torch.stack(full_cycles, dim=2)
        interleave = stacked.view(b, s * 3, f)

        interleave = interleave[:, :-1, :]
        true_actions = a[true_action_ids]

        return interleave, true_actions

    def update(self, R, s, a):
        interleave, true_actions = self.shape_X_y(R, s, a)
        self.ensemble.update(interleave, true_actions)

        raise Exception()

    def infer(self, R, s):
        self.ensemble.eval()
        pass
