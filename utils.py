from __future__ import print_function
import numpy as np
import torch
from torch.nn import functional as F


def normal_kl(mu1, lv1, mu2, lv2):
  # Both normals are isotropic; IWAE should not use this
  v1, v2 = lv1.exp(), lv2.exp()
  lstd1, lstd2 = lv1 / 2., lv2 / 2.
  kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
  return kl


def sigmoid_cross_entropy_with_logits(logits, labels):
  # TODO: Numerical stability
  p = torch.sigmoid(logits)
  return labels * torch.log(p + 1e-7) + (1 - labels) * torch.log(1 - p + 1e-7)


def normal_logpdf(x, mu, logvar):
  C = torch.Tensor([np.log(2. * np.pi)]).cuda()
  return (-.5 * (C + logvar) - (x - mu) ** 2. / (2. * torch.exp(logvar)))


def logmeanexp(x, dim=0):
  mmax, _ = torch.max(x, dim=dim, keepdim=True)
  return (torch.squeeze(mmax, dim=dim) +
          torch.log(torch.mean(torch.exp((x - mmax)), dim=dim)))
