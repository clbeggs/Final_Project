# Packages from CycleGAN repo: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
import sys
sys.path.append('../../../pytorch-CycleGAN-and-pix2pix/')
from models import create_model
from options.test_options import TestOptions
from util import util

import argparse

import torch
import torchvision

class CycleGan():
    def __init__(self, name, checkpoints_dir='../../../pytorch-CycleGAN-and-pix2pix/checkpoints/', epoch='latest'):
        # Setup sys.argv, since argparse is used in the CycleGAN repo
        self.addToSysArgv('dataroot', 'N/A')
        self.addToSysArgv('name', name)
        self.addToSysArgv('epoch', str(epoch))

        # Can (should?) change this to somewhere better
        self.addToSysArgv('checkpoints_dir', checkpoints_dir)
        self.addToSysArgv('model', 'cycle_gan')
        # uncomment to run on CPU
        # self.addToSysArgv('gpu_ids', '-1')

        opt = TestOptions().parse()
        self.model = create_model(opt)
        self.model.setup(opt) 

    @staticmethod
    def addToSysArgv(opt_name, value):
        sys.argv.append('--' + opt_name)
        sys.argv.append(value)

    def __call__(self, img):
        batch_img = torchvision.transforms.ToTensor()(img)
        batch_img = batch_img.reshape(1, *batch_img.shape)
        
        # Don't care about A->B, use all ones
        self.model.set_input({'B' : batch_img, 'A': torch.ones([1, 3, 256, 256]), 'A_paths' : 'N/A', 'B_paths' : 'N/A'})
        # Forward pass
        self.model.test()

        visuals = self.model.get_current_visuals()
        return util.tensor2im(visuals['fake_A'])
