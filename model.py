from __future__ import print_function
import argparse
import torch
from torch import nn
from torch.nn import functional as F

import utils


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

    self.latent_dim = latent_dim

  def encode(self, x):
    net = x
    net = F.relu(self.fc1(net))
    net = F.relu(self.fc2(net))
    return self.fc31(net), self.fc32(net)

  def reparameterize(self, mu, logvar, k=1):
    assert len(mu.size()) == len(logvar.size()) == 3
    latent_size = mu.size()
    std = torch.exp(0.5 * logvar)
    eps = torch.randn((k,) + latent_size[1:])
    eps = eps.cuda()
    return eps.mul(std).add_(mu)

  def decode(self, z):
    net = z
    net = F.relu(self.fc4(net))
    net = F.relu(self.fc5(net))
    return self.fc6(net)

  def forward(self, x, k=1):
    # Return everything with extra 0-th dim
    x = x.view(-1, 784)
    mu, logvar = self.encode(x)
    mu, logvar = mu.view((1,) + mu.size()), logvar.view((1,) + logvar.size())
    z = self.reparameterize(mu, logvar, k=k)
    x_logit = self.decode(z)
    assert x_logit.size() == (k,) + x.size()
    return x_logit, z, mu, logvar


def compute_elbo(x, x_logit, z, mu, logvar):
  # Assume everything has extra 0-th dim except x
  x = x.view(-1, 784)
  logpx = utils.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  zeros = torch.zeros(z.size()).cuda()
  logpz = utils.normal_logpdf(z, zeros, zeros)
  logqz = utils.normal_logpdf(z, mu, logvar)

  logpx = torch.sum(logpx, dim=-1)
  logpz = torch.sum(logpz, dim=-1)
  logqz = torch.sum(logqz, dim=-1)
  # First `logmeanexp` over sample dim, then `mean` over batch dim
  return torch.mean(utils.logmeanexp(logpx + logpz - logqz), dim=0)
