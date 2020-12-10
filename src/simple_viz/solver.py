import random
from typing import Tuple

import numpy as np
import torch

manual_seed = 999
torch.backends.cudnn.deterministic=True
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
np.random.seed(manual_seed)


class GANSolver():
    def __init__(self, gen, disc, lr, model):
        self.generator = gen
        self.discriminator = disc

        if model == 'DCGAN':
            self.gen_train_interval = 1
            self.noise_size = self.generator.noise_size
            self.DC = True
        else:
            self.gen_train_interval = 5
            self.noise_size = (256, 3)
            self.DC = False

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.discrim_optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.gen_optim = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))

        # Viz book keeping
        self.gen_loss = []
        self.disc_loss = []
        self.gen_grads = []
        self.disc_grads = []
        self.train_examples = []

    def get_plot_values(self):
        """Return loss, grad norms, and train examples for plotting"""
        return ((np.asarray(self.gen_loss), np.asarray(self.gen_grads)),
                (np.asarray(self.disc_loss), np.asarray(self.disc_grads)), self.train_examples)

    def store_gradient_norms(self) -> None:
        """Compute gradient norms and store in array
           Probably a better way to do this, but it works.
        """
        d_total_norm = 0
        g_total_norm = 0
        for d_param, g_param in zip(self.generator.parameters(), self.discriminator.parameters()):
            d_param_norm = d_param.grad.data.norm(2)
            d_total_norm += d_param_norm.item() ** 2
            g_param_norm = g_param.grad.data.norm(2)
            g_total_norm += g_param_norm.item() ** 2

        d_total_norm = d_total_norm ** (1. / 2)
        g_total_norm = g_total_norm ** (1. / 2)

        self.disc_grads.append(d_total_norm)
        self.gen_grads.append(g_total_norm)


    def store_recent_generated(self, generated_batch: torch.Tensor, epoch: int):
        """Store generated_batch for later visualization
            Args:
                generated_batch: Tensor containing most recent generated data
        """
        self.train_examples.append((generated_batch.cpu(), epoch))

    def get_noise(self, batch_size: int
                  ) -> torch.Tensor:
        """Return noise from normal distribution """
        if self.DC:
            return torch.randn(batch_size, self.noise_size, 1, 1)
        else:
            return torch.randn(batch_size, *self.noise_size)

    def train(self, dataloader: torch.utils.data.DataLoader,
              epochs: int,
              num_store: int = 10,
              ) -> None:
        """Train Generator and Discriminator
            Args:
                dataloader: torch DataLoader with data
                epochs: Number of epochs to run
                num_store: How many times to store relevant information
            Returns:
                gen_loss: Generator loss based on num_store
                disc_loss: Discriminator loss based on num_store
        """

        # Train Discrim more only for FC GAN
        for epoch in range(epochs):
            for i, batch in enumerate(dataloader):

                #####################
                # Train Discriminator
                #####################
                self.discriminator.zero_grad()
                disc_pred_real = self.discriminator(batch)
                disc_real_loss = self.criterion(disc_pred_real,
                                                torch.full(disc_pred_real.shape, fill_value=0.9))

                with torch.no_grad():
                    noise = self.get_noise(batch.shape[0])
                    generated_batch = self.generator(noise)

                disc_pred_fake = self.discriminator(generated_batch.detach())
                disc_fake_loss = self.criterion(disc_pred_fake, torch.zeros(disc_pred_fake.shape))

                disc_loss = disc_real_loss + disc_fake_loss
                disc_loss.backward()
                self.discrim_optim.step()

                # Train Generator on interval, as we want better discriminator
                # This is only applicable to FC GAN
                if i % self.gen_train_interval == 0:
                    ###################
                    # Train Generator
                    ###################
                    self.generator.zero_grad()

                    noise = self.get_noise(batch.shape[0])
                    generated_batch = self.generator(noise)

                    disc_pred_fake = self.discriminator(generated_batch)

                    gen_loss = self.criterion(disc_pred_fake,
                                              torch.full(disc_pred_fake.shape, fill_value=0.9))
                    gen_loss.backward()
                    self.gen_optim.step()

            # Print results on interval, and save recent generated_batch
            if epoch % 500 == 0:
                print("Epoch [%d] Gen Loss: [%f] Disc Loss: [%f]" % (epoch, gen_loss.item(), disc_loss.item()))
                self.store_recent_generated(generated_batch, epoch)

            # Record relevant stuff
            if epoch % num_store == 0:
                self.gen_loss.append(gen_loss.item())
                self.disc_loss.append(disc_loss.item())
                self.store_gradient_norms()
