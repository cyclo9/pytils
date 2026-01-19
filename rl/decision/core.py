from .utils import Interleaver

import torch.optim as optim
import torch.nn as nn


class DecisionModel(nn.Module):
    def __init__(self, model, state_size, d_model, lr=1e-3):
        super().__init__()
        self.model = model
        self.R_embed = nn.Linear(1, d_model).to(model.device)
        self.s_embed = nn.Linear(state_size, d_model).to(model.device)
        self.a_embed = nn.Linear(1, d_model).to(model.device)

        modules = [self.model, self.R_embed, self.s_embed, self.a_embed]
        params = [p for m in modules for p in m.parameters()]
        self.optimizer = optim.Adam(params, lr=lr)

    def embed(self, R, s, a):
        R_emb = self.R_embed(R)
        s_emb = self.s_embed(s)
        a_emb = self.a_embed(a)
        return R_emb, s_emb, a_emb

    def forward(self, R, s, a, S):
        R_emb, s_emb, a_emb = self.embed(R, s, a)

        core = Interleaver(R_emb, s_emb, a_emb, S)
        X = core.stack().pair().drop_tail()
        ids = core.to_ids().pair().take_tail()

        out = self.model(X)
        target = a[ids]
        return out.squeeze(1), target.squeeze(1).long()

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
