

# How to train:
#### FCGAN:
FCGAN: `python simple_viz.py --model=FCGAN --num_examples=3 --epochs=2000 --train`

DCGAN: `python simple_viz.py --model=DCGAN --num_examples=3 --epochs=18000 --train`

# How to visualize:
DCGAN: `python simple_viz.py --model=DCGAN`


FCGAN: `python simple_viz.py --model=FCGAN`

# Code Overview

```
simple_viz/
    ├── __init__.py
    ├── simple_viz.py       - main function that trains/displays network
    ├── data.py             - pytorch dataset
    ├── models.py           - FCGAN and DCGAN model implementations
    ├── solver.py           - for training GAN
    ├── utils.py            - plt utils
```
