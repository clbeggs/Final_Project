import argparse
import os
import random
import sys

import numpy as np
import torch

from data import PointCloudDataset
from models import (DC_Discriminator, DC_Generator, SimpleVizDiscriminator,
                    SimpleVizGenerator)
from solver import GANSolver
from utils import plot_data, plot_training_result

manual_seed = 7
torch.backends.cudnn.deterministic = True
torch.manual_seed(manual_seed)
random.seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
np.random.seed(manual_seed)


def weights_init(model):
    """Define own model weight initialization,

        Reference: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#discriminator
    """
    classname = model.__class__.__name__
    if (classname.find('Conv') != -1) and (classname.find("torch") != -1):
        torch.nn.init.normal_(model.weight, 0.0, 0.02)

    elif (classname.find('BatchNorm') !=
          -1) and (classname.find("torch") != -1):
        torch.nn.init.normal_(model.weight, 1.0, 0.02)
        torch.nn.init.zeros_(model.bias)

    elif classname.find("Layer") != -1:
        for child in model.named_children():
            for param in child[1]:
                if param.__class__.__name__.find("Conv") != -1:
                    torch.nn.init.normal_(param.weight, 0.0, 0.02)
                if param.__class__.__name__.find("BatchNorm") != -1:
                    torch.nn.init.normal_(param.weight, 1.0, 0.02)
                    torch.nn.init.zeros_(param.bias)


def main(opts):
    # Choose device
    if opts.forcecpu:
        device = "cpu"
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # Init dataloader
    data = PointCloudDataset(opts.data_type, opts.num_examples)
    loader = torch.utils.data.DataLoader(
        dataset=data, batch_size=opts.batchsize, shuffle=True
        )

    # Init Models
    if opts.model == "FCGAN":
        gen = SimpleVizGenerator(dim_h=opts.gen_hid_size)
        disc = SimpleVizDiscriminator(dim_h=opts.disc_hid_size)
        solver = GANSolver(gen, disc, lr=0.001, model=opts.model)

    else:
        gen = DC_Generator()
        disc = DC_Discriminator()
        gen.apply(weights_init)
        disc.apply(weights_init)
        solver = GANSolver(gen, disc, lr=0.0002, model=opts.model)

    if opts.train is False:
        cwd = os.path.abspath('')
        filename = cwd + "/model_weights/" + opts.model

        try:
            gen.load_state_dict(torch.load(filename + "_gen"))
            disc.load_state_dict(torch.load(filename + "_disc"))

        except FileNotFoundError:
            print("Need to train the models before visualizing.")
            sys.exit(1)
        plot_data(solver)

    else:
        print("TRAINING--------")
        print("DATATYPE: ", opts.data_type)
        print("EPOCHS: ", opts.epochs)
        print("BATCH SIZE: ", opts.batchsize)
        print("CUDA: ", torch.cuda.current_device())
        print("-----------------\n\n")

        # Train model!
        solver.train(dataloader=loader, epochs=opts.epochs)

        # Plot Results
        plot_training_result(solver)

    # Save weights
    if opts.save_weights:
        if not os.path.exists("model_weights"):
            os.makedirs('model_weights')
        cwd = os.path.abspath('')
        filename = cwd + "/model_weights/" + opts.model
        torch.save(gen.state_dict(), filename + "_gen")
        torch.save(disc.state_dict(), filename + "_disc")


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='FCGAN, DCGAN')
    parser.add_argument('--epochs', required=False, default=18000, type=int)
    parser.add_argument('--batchsize', required=False, default=20, type=int)
    parser.add_argument('--num_examples', required=False, default=3, type=int)
    parser.add_argument('--data_type', required=False, default="channel_gauss")
    parser.add_argument('--forcecpu', required=False, default=False)
    parser.add_argument('--train', required=False, action='store_true')
    parser.add_argument('--gen_hid_size', required=False, default=256, type=int)
    parser.add_argument('--disc_hid_size', required=False, default=128, type=int)
    parser.add_argument('--save_weights', required=False, default=True, type=bool)

    opts = parser.parse_args()
    if (opts.data_type == "channel_gauss") and (opts.model == "FCGAN"):
        raise Exception("Can't use channel_gauss datatype with FCGAN")
    main(opts)
