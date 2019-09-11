from __future__ import print_function
import argparse
import math
import torch
import torch.distributions as dist
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):

  def __init__(self, latent_dim=20):
    super(VAE, self).__init__()

    self.fc1 = nn.Linear(784, 400)
    self.fc2 = nn.Linear(400, 400)
    self.fc31 = nn.Linear(400, latent_dim)
    self.fc32 = nn.Linear(400, latent_dim)
    self.fc4 = nn.Linear(latent_dim, 400)
    self.fc5 = nn.Linear(400, 400)
    self.fc6 = nn.Linear(400, 784)

  def encode(self, x):
    net = x
    net = F.relu(self.fc1(net))
    net = F.relu(self.fc2(net))
    return self.fc31(net), self.fc32(net).exp()

  def decode(self, z):
    net = z
    net = F.relu(self.fc4(net))
    net = F.relu(self.fc5(net))
    return self.fc6(net)

  def forward(self, x, k=1):
    x = x.view(-1, 784)
    qz_x = dist.Normal(*self.encode(x))
    z = qz_x.rsample(torch.Size([k]))
    px_z = dist.Bernoulli(logits=self.decode(z))
    return qz_x, px_z, z


def compute_elbo(x, qz_x, px_z, z):
  x = x.view(-1, 784)
  lpx_z = px_z.log_prob(x).sum(-1)
  lpz = dist.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(-1)
  lqz_x = qz_x.log_prob(z).sum(-1)

  lw = lpz + lpx_z - lqz_x
  return (torch.logsumexp(lw, 0) - math.log(lw.size(0))).mean(0)


def compute_elbo_dreg(x, qz_x, px_z, z):
  x = x.view(-1, 784)
  lpx_z = px_z.log_prob(x).sum(-1)
  lpz = dist.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(-1)
  qz_x_ = qz_x.__class__(qz_x.loc.detach(), qz_x.scale.detach())
  lqz_x = qz_x_.log_prob(z).sum(-1)

  lw = lpz + lpx_z - lqz_x
  with torch.no_grad():
    reweight = torch.exp(lw - torch.logsumexp(lw, 0))
    z.register_hook(lambda grad: reweight.unsqueeze(-1) * grad)

  return (reweight * lw).sum(0).mean(0)
