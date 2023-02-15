import random
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from sklearn.linear_model import LogisticRegression


EPS = 1e-15

class SugbCon(torch.nn.Module):

    def __init__(self, hidden_channels, encoder, pool, scorer):
        super(SugbCon, self).__init__()
        self.encoder = encoder
        self.hidden_channels = hidden_channels
        self.pool = pool
        self.scorer = scorer
        self.marginloss = nn.MarginRankingLoss(0.5)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()
        self.tau: float = 0.9
        self.beta = 1
        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)

    def reset_parameters(self):
        reset(self.scorer)
        reset(self.encoder)
        reset(self.pool)

    def forward(self, x, edge_index):
        r""" Return node and subgraph representations of each node before and after being shuffled """
        hidden = self.encoder(x, edge_index)
        '''if index is None:
            return hidden
        z = hidden[index]
        summary = self.pool(hidden, edge_index, batch)
        return z, summary'''
        return hidden

    def loss(self, hidden1, summary1):
        r"""Computes the margin objective."""

        shuf_index = torch.randperm(summary1.size(0))

        hidden2 = hidden1[shuf_index]
        summary2 = summary1[shuf_index]

        logits_aa = torch.sigmoid(torch.sum(hidden1 * summary1, dim = -1))
        logits_bb = torch.sigmoid(torch.sum(hidden2 * summary2, dim = -1))
        logits_ab = torch.sigmoid(torch.sum(hidden1 * summary2, dim = -1))
        logits_ba = torch.sigmoid(torch.sum(hidden2 * summary1, dim = -1))

        TotalLoss = 0.0
        ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
        TotalLoss += self.marginloss(logits_aa, logits_ba, ones)
        TotalLoss += self.marginloss(logits_bb, logits_ab, ones)

        return TotalLoss

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def contrastive_loss_wo_cross_view(self,h1, h2, z):
        f = lambda x: torch.exp(x)
        cross_sim = f(self.sim(h1, z))
        return -torch.log(cross_sim.diag() / cross_sim.sum(dim=-1))

    def loss1(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):

        h1 = self.projection(z1)
        h2 = self.projection(z2)
        l1 = self.beta*self.semi_loss(h1, h2)+(1-self.beta)*self.contrastive_loss_wo_cross_view(h1,h2,h2)
        l2 = self.beta*self.semi_loss(h2, h1)+(1-self.beta)*self.contrastive_loss_wo_cross_view(h2,h1,h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def margin_loss(self, z1, z2):
        logits_aa = torch.sigmoid(torch.sum(z1, dim=-1))

        logits_ba = torch.sigmoid(torch.sum(z2, dim=-1))

        ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
        TotalLoss = self.marginloss(logits_aa, logits_ba, ones)

        return TotalLoss
    def loss2(self,z1,z2):
        z1 = torch.sigmoid((torch.sum(z1,dim=-1)))
        z2 = torch.sigmoid((torch.sum(z2,dim=-1)))
        z1 = z1.unsqueeze(0)
        z2 = z2.unsqueeze(0)
        logits = torch.cat((z1, z2), 1)
        b_xent = nn.BCEWithLogitsLoss()
        lbl_1 = torch.ones(1, z1.size(1))
        lbl_2 = torch.ones(1, z1.size(1))
        lbl = torch.cat((lbl_1,lbl_2),1).cuda()
        loss = b_xent(logits, lbl)
        return loss


    def test(self, train_z, train_y, val_z, val_y, test_z, test_y, solver='lbfgs',
             multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream task."""
        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        val_acc = clf.score(val_z.detach().cpu().numpy(), val_y.detach().cpu().numpy())
        test_acc = clf.score(test_z.detach().cpu().numpy(), test_y.detach().cpu().numpy())
        return val_acc, test_acc