# CSCI-4622 Final Project

**Team Name:** A Cool Name

**Group Members:** Connor Thompson, Soroush Khadem, Greg Lund, Chris Beggs

---

### Downloading Weights:
To download all of the weights in the correct directory, from the main directory of this repo run the following
```bash
bash scripts/download_models.sh
```
To download weights for an individual model, from the main directory run the following:
```bash
bash scripts/single_download.sh <model_name>
```
model_name can be any of the following:
- pix2pix_dogs
- pix2pix_pizza
- pix2pix_trees
- pix2pix_apples
- cyclegan_pizza
- cyclegan_trees
- cyclegan_quickdraw_trees
- cyclegan_apples

### Important Dates:
December 7th: Final Presentation <br/>
December 9th: Final report due

### Relavant Links:
Original paper: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)


Tensorflow demo: https://www.tensorflow.org/tutorials/generative/pix2pix

Interactive demo: https://affinelayer.com/pixsrv/

UMich DL Course Slides: https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture20.pdf

GAN Tutorial(Goodfellow): [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160)

Tips and Tricks to make GANs work: https://github.com/soumith/ganhacks

More stable GAN objective function:
 [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf) and 
 [Improved training of Wasserstein GAN](https://arxiv.org/abs/1704.00028)

 Related latent space paper: [Interpreting the Latent Space of GANs for Semantic Face Editing](https://arxiv.org/pdf/1907.10786.pdf)



