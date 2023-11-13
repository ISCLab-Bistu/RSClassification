import argparse
import os
import numpy as np
import math

import pandas as pd
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from rmsm.datasets.rampy import normalise

os.makedirs("spectrum", exist_ok=True)

img_shape = (1, 815)

device = 'cuda'
cuda = True if torch.cuda.is_available() else False


# raman dataset
class RamanDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        result = self.data[index]
        label = torch.tensor(result[0]).to(device)
        spectrum = normalise(y=result[1:], x=0, method='minmax')
        spectrum = torch.from_numpy(spectrum).float().to(device)
        return spectrum, label


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(815, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

batch_size = 64
n_epochs = 300
# build data loader
raman_data = pd.read_csv('./single_cell/results/single_cell.csv').iloc[1:200, 1:].values
raman_dataset = RamanDataset(raman_data)
dataloader = DataLoader(raman_dataset, batch_size=batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.01, betas=(0.5, 0.99))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.01, betas=(0.5, 0.99))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(n_epochs):
    for i, (spectrum, _) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(Tensor(spectrum.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(spectrum.size(0), 1).fill_(0.0), requires_grad=False)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = torch.randn(spectrum.shape[0], spectrum.shape[1]).to(device)
        # print(z.shape)

        # Generate a batch of spectrum
        gen_spectrum = generator(z)
        # print(gen_spectrum.shape)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_spectrum), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(spectrum), valid)
        fake_loss = adversarial_loss(discriminator(gen_spectrum.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        # if batches_done % opt.sample_interval == 0:
        #     save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

# 
noise = torch.randn(200, img_shape[1]).to(device)
print(noise)
generator_data = generator(noise)
print(generator_data)
data_numpy = torch.squeeze(generator_data).detach().cpu()
data_df = pd.DataFrame(data_numpy.numpy())
data_df.to_csv('gan_raman_data.csv', index=False)
