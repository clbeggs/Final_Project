import random

import numpy as np
import torch
from scipy.stats import multivariate_normal

manual_seed = 7
torch.backends.cudnn.deterministic=True
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
np.random.seed(manual_seed)

def get_data(data_type='gaussian', N=10):
    low = 1
    high = 100
    ### Torus Eq: (c - sqrt{x^2 + y^2})^2 + z^2 = a^2
    ### Cone Eq: (x^2/ a^2) + (y^2/b^2) = (z^2/c^2)
    ### Hyperboloid One Sheet Eq: (x^2/ a^2) + (y^2/b^2) - (z^2/c^2) = 1
    ### Saddle Eq: (x^2/ a^2) - (y^2/b^2) = (z/c)

    ## Spiral Eq. (tcos(6t), tsin(6t), t)
    ## pair  (1.5*tcos(6t), 1.5*tsin(6t), t)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = []
    if data_type == 'spiral':
        for _ in range(N):
            t = np.random.uniform(low=low, high=high, size=1)[0]
            data.append([t * np.cos(10 * t), t * np.sin(10 * t), t])

    elif data_type == 'gaussian':
        rv = multivariate_normal(mean=[0, 0], cov=[[1.0, 0.0], [0.0, 1.0]])
        x, y = np.mgrid[-2:2:.1, -2:2:.1]
        pos = np.dstack((x, y))
        z = rv.pdf(pos)
        positions = np.vstack([x.ravel(), y.ravel(), z.ravel()])
        data = positions.T  # Shape (1600, 3)
        return torch.from_numpy(data).float().to(device)

    elif data_type == 'channel_gauss':
        data = np.empty((N, 1, 64, 64))  # DCGAN requires 64x64 input
        for i in range(N):
            rv = multivariate_normal(mean=[0, 0], cov=[[i+1, 0.0], [0.0, i+1]])
            x, y = np.mgrid[-2:2:.0625, -2:2:.0625]
            pos = np.dstack((x, y))
            z = rv.pdf(pos)
            data[i] = z
        return torch.from_numpy(data).float().to(device)

    elif data_type == 'full_gaussian':
        a = np.random.uniform(.1, 2)
        b = np.random.uniform(.1, 2)
        data = np.empty((N, 256, 3))

        for i in range(N):
            rv = multivariate_normal(mean=[0, 0], cov=[[a, 0.0], [0.0, b]])
            x, y = np.mgrid[-2:2:.25, -2:2:.25]
            pos = np.dstack((x, y))
            z = rv.pdf(pos)
            positions = np.vstack([x.ravel(), y.ravel(), z.ravel()])
            data[i] = positions.T  # Shape (256, 3)

        return torch.from_numpy(data).float().to(device)


    data = np.array(data)
    data = (data / np.linalg.norm(data, ord=2))
    return torch.from_numpy(data).float().to(device)


class PointCloudDataset(torch.utils.data.Dataset):
    """Pytorch Dataset class for Dataloader
    """
    def __init__(self, data_type='channel_gauss', N=2):
        self.data = get_data(data_type, N)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
