import random

import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.nn as nn
import torch

from ..builder import PIPELINES


@PIPELINES.register_module()
class AddNoise(object):
    def __init__(self, num=None, noise_std=0.01, **kwargs):
        self.kwargs = kwargs
        self.num_new_data = num
        self.noise_std = noise_std

    def __call__(self, results):
        print("AddNoise")
        labels = results['labels']
        spectrum = results['spectrum']
        num_new_data = len(labels)
        length_label = len(labels)
        if self.num_new_data is not None:
            num_new_data = self.num_new_data

        indices = list(range(length_label))
        random.shuffle(indices)

        result_spectrum = []
        result_label = []
        for i in range(length_label):
            result_spectrum.append(spectrum[i])
            result_label.append(labels[i])
        for i in range(num_new_data):
            if i >= length_label:
                k = indices[i - length_label]
            else:
                k = indices[i]
            original_spectrum = spectrum[k]
            noise_std = self.noise_std * np.std(original_spectrum)
            noise = np.random.normal(scale=noise_std, size=original_spectrum.shape)
            new_data = original_spectrum + noise
            result_spectrum.append(new_data)
            result_label.append(labels[k])

        result_spectrum = np.array(result_spectrum)
        result_label = np.array(result_label)
        results['spectrum'] = result_spectrum
        results['labels'] = result_label
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'num_new_data={self.num_new_data}, '
                    f'noise_std={self.noise_std})')
        return repr_str


@PIPELINES.register_module()
class IntensityFactory(object):
    def __init__(self, num=None, **kwargs):
        self.kwargs = kwargs
        self.num_new_data = num

    def __call__(self, results):
        print("IntensityFactory")
        labels = results['labels']
        spectrum = results['spectrum']
        num_new_data = len(labels)
        length_label = len(labels)
        if self.num_new_data is not None:
            num_new_data = self.num_new_data

        indices = list(range(length_label))  # 0-length
        random.shuffle(indices)  # 

        result_spectrum = []
        result_label = []
        for i in range(length_label):
            result_spectrum.append(spectrum[i])
            result_label.append(labels[i])
        for i in range(num_new_data):
            if i >= length_label:
                k = indices[i - length_label]
            else:
                k = indices[i]
            original_spectrum = spectrum[k]
            intensity_factor = np.random.uniform(0.2, 2)
            new_data = original_spectrum * intensity_factor
            result_spectrum.append(new_data)
            result_label.append(labels[k])

        result_spectrum = np.array(result_spectrum)
        result_label = np.array(result_label)
        results['spectrum'] = result_spectrum
        results['labels'] = result_label
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'num_new_data={self.num_new_data})')
        return repr_str


# raman dataset
class RamanDataset(Dataset):
    def __init__(self, labels, spectrum):
        self.labels = labels
        self.spectrum = spectrum

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        spectrum = torch.from_numpy(self.spectrum).float()
        return spectrum[index], self.labels[index]


class Generator(nn.Module):
    def __init__(self, ram_shape):
        super(Generator, self).__init__()
        self.ram_shape = ram_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.ram_shape[1], 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.ram_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        ram = self.model(z)
        ram = ram.view(ram.size(0), *self.ram_shape)
        return ram


class Discriminator(nn.Module):
    def __init__(self, ram_shape):
        super(Discriminator, self).__init__()
        self.ram_shape = ram_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.ram_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, ram):
        ram_flat = ram.view(ram.size(0), -1)
        validity = self.model(ram_flat)

        return validity


@PIPELINES.register_module()
class GANRaman(object):
    def __init__(self, train_label=0, num=None, batch_size=256, n_epochs=1000, adversarial_loss=torch.nn.BCELoss(),
                 device='cuda', **kwargs):
        self.kwargs = kwargs
        self.num_new_data = num
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.train_label = int(train_label)
        self.device = device
        # Loss function
        self.adversarial_loss = adversarial_loss

    def __call__(self, results):
        print("GANRaman")
        labels = results['labels']
        spectrum = results['spectrum']

        result_spectrum = []
        result_label = []
        length_label = len(labels)
        for i in range(length_label):
            result_spectrum.append(spectrum[i])
            result_label.append(labels[i])

        indices = np.where(labels == self.train_label)[0]

        labels = labels[indices]
        spectrum = spectrum[indices]

        # According to the train_label to get the specified data set for training
        num_new_data = len(labels)
        ram_shape = (1, spectrum.shape[1])
        # Initialize generator and discriminator
        self.generator = Generator(ram_shape)
        self.discriminator = Discriminator(ram_shape)
        if self.num_new_data is not None:
            num_new_data = self.num_new_data

        raman_dataset = RamanDataset(labels, spectrum)
        dataloader = DataLoader(raman_dataset, batch_size=self.batch_size, shuffle=True)
        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.001, betas=(0.5, 0.99))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.001, betas=(0.5, 0.99))

        # Define a learning rate scheduler
        num_epochs = self.n_epochs
        scheduler_G = CosineAnnealingLR(optimizer_G, T_max=num_epochs)
        scheduler_D = CosineAnnealingLR(optimizer_D, T_max=num_epochs)

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.adversarial_loss.to(self.device)
        if self.device == 'cuda':
            cuda = True
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        for epoch in range(self.n_epochs):
            for i, (spectrum, _) in enumerate(dataloader):
                # Adversarial ground truths
                valid = Variable(Tensor(spectrum.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(spectrum.size(0), 1).fill_(0.0), requires_grad=False)
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = torch.randn(spectrum.shape[0], spectrum.shape[1]).to(self.device)
                # print(z.shape)

                # Generate a batch of spectrum
                gen_spectrum = self.generator(z)
                # print(gen_spectrum.shape)

                # Loss measures generator's ability to fool the discriminator
                g_loss = self.adversarial_loss(self.discriminator(gen_spectrum), valid)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                spectrum = spectrum.to(self.device)
                real_loss = self.adversarial_loss(self.discriminator(spectrum), valid)

                fake_loss = self.adversarial_loss(self.discriminator(gen_spectrum.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                scheduler_G.step()
                scheduler_D.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

        noise = torch.randn(num_new_data, ram_shape[1]).to(self.device)
        generator_data = self.generator(noise)
        new_label = [self.train_label] * num_new_data
        generator_data = torch.squeeze(generator_data).detach().cpu().tolist()

        result_spectrum = result_spectrum + generator_data
        result_label = result_label + new_label

        result_spectrum = np.array(result_spectrum)
        result_label = np.array(result_label)

        results['spectrum'] = result_spectrum
        results['labels'] = result_label
        return results
