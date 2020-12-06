import sys
sys.path.append('../../pytorch-CycleGAN-and-pix2pix/')
from models import create_model
from options.test_options import TestOptions
from util import util
import argparse
import torch
import torchvision

class Pix2Pix():
    def __init__(self,model_name,epoch='latest',use_gpu=True):

        network_type = model_name.split('_')[0]
        self.addToSysArgv('dataroot','N/A')
        self.addToSysArgv('name',model_name)
        self.addToSysArgv('epoch',str(epoch))
        self.addToSysArgv('checkpoints_dir', '../checkpoints')
        self.addToSysArgv('model', network_type)
        self.addToSysArgv('gpu_ids','0') if use_gpu else self.addToSysArgv('gpu_ids','-1')

        opt = TestOptions().parse()
        self.model = create_model(opt)
        self.model.setup(opt)

    def __call__(self, img):
        batch_img = torchvision.transforms.ToTensor()(img)
        batch_img = batch_img.reshape(1, *batch_img.shape)
        
        # Don't care about A->B, use all ones
        self.model.set_input({'B' : batch_img, 'A': torch.ones([1, 3, 256, 256]), 'A_paths' : 'N/A', 'B_paths' : 'N/A'})
        # Forward pass
        self.model.test()

        visuals = self.model.get_current_visuals()
        return util.tensor2im(visuals['fake_B'])

    @staticmethod
    def addToSysArgv(opt_name, value):
        sys.argv.append('--' + opt_name)
        sys.argv.append(value)
        
