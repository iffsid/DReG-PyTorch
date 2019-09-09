from __future__ import print_function
import argparse
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import model as model_


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train-k', type=int, default=1,
                    help='number of iwae samples during training')
parser.add_argument('--test-k', type=int, default=1,
                    help='number of iwae samples during testing')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='learning rate for Adam')
parser.add_argument('--latent-dim', type=int, default=20,
                    help='number of latent variables')
parser.add_argument('--dreg', action='store_true',
                    help='use DReG for inference network gradients')
parser.add_argument('--ablation', action='store_true')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


model = model_.VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


def train(epoch):
  model.train()
  train_loss = 0
  for batch_idx, (data, _) in enumerate(train_loader):
    data = data.to(device)
    # Dynamic binarization
    sample = torch.rand(data.size()).to(device)
    data = (sample < data).float()

    optimizer.zero_grad()
    x_logit, z, mu, logvar = model(data, k=args.train_k)
    loss = -model_.compute_elbo_dreg(data, x_logit, z, mu, logvar) if args.dreg \
      else -model_.compute_elbo(data, x_logit, z, mu, logvar)
    loss.backward()
    train_loss += -loss.item()
    optimizer.step()

    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader),
          loss.item()))

  avg_loss = train_loss / (batch_idx + 1)
  print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, avg_loss))

  return avg_loss


def test(epoch):
  model.eval()
  test_loss = 0
  with torch.no_grad():
    for i, (data, _) in enumerate(test_loader):
      data = data.to(device)
      x_logit, z, mu, logvar = model(data, k=args.test_k)
      test_loss += model_.compute_elbo(
          x=data, x_logit=x_logit, z=z, mu=mu, logvar=logvar).item()
      if i == 0:
        n = min(data.size(0), 8)
        comparison = torch.cat(
            [data[:n],
             x_logit.view(args.batch_size * args.test_k, 1, 28, 28)[:n]])
        save_image(comparison.cpu(),
                   './results/reconstruction_' + str(epoch) + '.png', nrow=n)

  test_loss /= (i + 1)
  print('====> Test set loss: {:.4f}'.format(test_loss))
  return test_loss


def main():
  train_stats, test_stats = [], []
  for epoch in range(1, args.epochs + 1):
    train_stats.append(train(epoch))
    test_stats.append(test(epoch))
    with torch.no_grad():
      sample = torch.randn(64, 20).to(device)
      sample = model.decode(sample).cpu()
      save_image(sample.view(64, 1, 28, 28),
                 './results/sample_' + str(epoch) + '.png')

  return train_stats, test_stats


if __name__ == '__main__':
  if not os.path.exists('./results'):
    os.makedirs('./results')

  if args.ablation:
    # Compare usual iwae with dreg iwae
    args.dreg = False
    regular_train, regular_test = main()

    args.dreg = True
    dreg_train, dreg_test = main()

    # Compare curve
    plt.figure()
    x = list(range(1, args.epochs + 1))
    plt.plot(x, regular_train, label='Regular')
    plt.plot(x, dreg_train, label='DReG')
    plt.title('training time iwae elbo bound (k={})'.format(args.train_k))
    plt.ylabel("ELBO")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig('./results/training_curve.png')

    plt.figure()
    plt.plot(x, regular_test, label='Regular')
    plt.plot(x, dreg_test, label='DReG')
    plt.title('test time iwae elbo bound (k={})'.format(args.test_k))
    plt.ylabel("ELBO")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig('./results/test_curve.png')
  else:
    main()
