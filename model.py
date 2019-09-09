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

    self.generator_layers = [self.fc1, self.fc2, self.fc31, self.fc32]
    self.inference_layers = [self.fc4, self.fc5, self.fc6]

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
    # Return tensors have an extra 0-th dim
    x = x.view(-1, 784)
    mu, logvar = self.encode(x)
    mu, logvar = mu[None, ...], logvar[None, ...]
    z = self.reparameterize(mu, logvar, k=k)
    x_logit = self.decode(z)
    assert x_logit.size() == (k,) + x.size()
    return x_logit, z, mu, logvar

  def freeze_inference(self):
    _freeze_layers(self.inference_layers)

  def freeze_generator(self):
    _freeze_layers(self.generator_layers)

  def unfreeze_inference(self):
    _unfreeze_layers(self.inference_layers)

  def unfreeze_generator(self):
    _unfreeze_layers(self.generator_layers)


def _freeze_layers(ls):
  for l in ls:
    for p in l.parameters():
      p.requires_grad = False


def _unfreeze_layers(ls):
  for l in ls:
    for p in l.parameters():
      p.requires_grad = True


def compute_elbo(x, x_logit, z, mu, logvar):
  # Assume all input tensors have extra 0-th dim except x
  x = x.view(-1, 784)
  logpx = utils.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  zeros = torch.zeros(z.size()).cuda()
  logpz = utils.normal_logpdf(z, zeros, zeros)
  logqz = utils.normal_logpdf(z, mu, logvar)

  logpx = torch.sum(logpx, dim=-1)
  logpz = torch.sum(logpz, dim=-1)
  logqz = torch.sum(logqz, dim=-1)
  # First `logmeanexp` over sample dim, then `mean` over batch dim
  return torch.mean(utils.logmeanexp(logpx + logpz - logqz, dim=0), dim=0)


def compute_elbo_dreg(x, x_logit, z, mu, logvar):
  # Call this function for inference network gradients
  # Assume all input tensors have extra 0-th dim except x
  x = x.view(-1, 784)
  logpx = utils.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  zeros = torch.zeros(z.size()).cuda()
  logpz = utils.normal_logpdf(z, zeros, zeros)

  # Gradient only go to \phi through z
  mu_stop, logvar_stop = mu.detach(), logvar.detach()

  logqz = utils.normal_logpdf(z, mu_stop, logvar_stop)

  logpx = torch.sum(logpx, dim=-1)
  logpz = torch.sum(logpz, dim=-1)
  logqz = torch.sum(logqz, dim=-1)

  logw = logpx + logpz - logqz
  assert len(logw.size()) == 2

  # Don't backprop through w_i / (\sum_i w_i)
  with torch.no_grad():
    reweight = torch.exp(
        logw - torch.logsumexp(logw, dim=0, keepdim=True)).pow(2)

  return torch.mean(torch.sum(reweight * logw, dim=0), dim=0)
