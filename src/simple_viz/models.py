import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimpleVizDiscriminator(torch.nn.Module):
    def __init__(self,
                 dim_in=3,
                 dim_h=3,
                 ) -> None:
        super(SimpleVizDiscriminator, self).__init__()

        self.distrib = torch.distributions.Uniform(low=0, high=1)
        self.Discriminator = torch.nn.Sequential(torch.nn.Linear(dim_in, dim_h),
                                                 torch.nn.LeakyReLU(0.01, inplace=True),

                                                 torch.nn.Linear(dim_h, dim_h),
                                                 torch.nn.LeakyReLU(0.01, inplace=True),

                                                 torch.nn.Linear(dim_h, 1),
                                                 # torch.nn.Sigmoid()
                                                 )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Forward pass of Discriminator
            Args:
                latent: torch.Tensor, dtype: torch.float, shape: (N, dim_out) generated image
            Returns:
                output: torch.Tensor, dtype: torch.float, shape: (N,1) discriminator prediction
        """
        output = self.Discriminator(latent)
        return output

class SimpleVizGenerator(torch.nn.Module):
    def __init__(self,
                 dim_in=3,
                 dim_h=3,
                 dim_out=3
                 ) -> None:
        super(SimpleVizGenerator, self).__init__()
        self.distrib = torch.distributions.Normal(loc=0, scale=1)

        self.Generator = torch.nn.Sequential(torch.nn.Linear(dim_in, dim_h),
                                             torch.nn.LeakyReLU(0.01, inplace=True),

                                             torch.nn.Linear(dim_h, dim_h),
                                             torch.nn.LeakyReLU(0.01, inplace=True),

                                             torch.nn.Linear(dim_h, dim_out),
                                             torch.nn.Tanh(),
                                             )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Simple GAN
            Args:
                x: torch.Tensor, dtype: torch.float, shape: (N, sample_size, 3) input examples with batch size N
            Returns:
                latent: torch.Tensor, dtype: torch.float, shape: (N, dim_out) generated image
                output: torch.Tensor, dtype: torch.float, shape: (1,) discriminator prediction
        """
        latent = self.Generator(x)
        return latent


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 batch_norm: bool
                 ) -> None:
        """Conv block for DCGAN"""
        super(ConvLayer, self).__init__()
        self.conv_layer = torch.nn.Sequential()
        self.conv_layer.add_module("conv",
                                   torch.nn.Conv2d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=4,
                                                   stride=2,  # from architecture guidelines in DCGAN paper
                                                   padding=1,
                                                   bias=False))
        if batch_norm:
            self.conv_layer.add_module("batch",
                                       torch.nn.BatchNorm2d(num_features=out_channels))

        self.conv_layer.add_module("leakyrelu",
                                   torch.nn.LeakyReLU(negative_slope=0.2,
                                                      inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of DCGAN conv block"""
        return self.conv_layer(x)

class DC_Discriminator(torch.nn.Module):
    """DCGAN Discriminator
       Reference:
            https://arxiv.org/pdf/1511.06434.pdf
    """

    def __init__(self, in_channels: int = 1,
                 hid_channels: int = 16
                 ) -> None:
        super(DC_Discriminator, self).__init__()

        # Bias is set to false as batch norm nulls it
        # Ref: https://discuss.pytorch.org/t/any-purpose-to-set-bias-false-in-densenet-torchvision/22067
        self.model = torch.nn.Sequential(
            ConvLayer(in_channels=in_channels,
                      out_channels=hid_channels,
                      batch_norm=False),

            ConvLayer(in_channels=hid_channels,
                      out_channels=hid_channels * 2,
                      batch_norm=True),

            ConvLayer(in_channels=hid_channels * 2,
                      out_channels=hid_channels * 4,
                      batch_norm=True),

            ConvLayer(in_channels=hid_channels * 4,
                      out_channels=hid_channels * 8,
                      batch_norm=True),

            torch.nn.Conv2d(in_channels=hid_channels * 8,
                            out_channels=1,
                            kernel_size=4,
                            stride=1,
                            padding=0,
                            bias=False),
            # torch.nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of DCGAN Discriminator"""
        pred = self.model(x)
        return pred.view(-1, 1).squeeze(1)


class ConvTransposeLayer(torch.nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 batch_norm: bool
                 ) -> None:
        """Conv block for DCGAN"""
        super(ConvTransposeLayer, self).__init__()

        self.conv_layer = torch.nn.Sequential()
        self.conv_layer.add_module("convT",
                                   torch.nn.ConvTranspose2d(in_channels=in_channels,
                                                            out_channels=out_channels,
                                                            kernel_size=4,
                                                            stride=2,  # from architecture guidelines in DCGAN paper
                                                            padding=1,
                                                            bias=False))
        if batch_norm:
            self.conv_layer.add_module("batch",
                                       torch.nn.BatchNorm2d(num_features=out_channels))
        self.conv_layer.add_module("relu",
                                   torch.nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of DCGAN conv block"""
        return self.conv_layer(x)


class DC_Generator(torch.nn.Module):
    """DCGAN Generator,
        Note: Deconvolution \equiv ConvTranspose
       Reference:
            https://arxiv.org/pdf/1511.06434.pdf
    """
    def __init__(self, in_channels: int = 32,
                 hid_channels: int = 16,
                 out_channels: int = 1
                 ) -> None:
        super(DC_Generator, self).__init__()
        self.noise_size = in_channels  # For Solver

        # Bias is set to false as batch norm nulls it
        # Ref: https://discuss.pytorch.org/t/any-purpose-to-set-bias-false-in-densenet-torchvision/22067
        self.model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=in_channels,
                                     out_channels=hid_channels * 8,
                                     kernel_size=4,
                                     stride=1,
                                     padding=0,
                                     bias=False),
            torch.nn.BatchNorm2d(num_features=hid_channels * 8),
            torch.nn.ReLU(inplace=True),

            ConvTransposeLayer(in_channels=hid_channels * 8,
                               out_channels=hid_channels * 4,
                               batch_norm=True),

            ConvTransposeLayer(in_channels=hid_channels * 4,
                               out_channels=hid_channels * 2,
                               batch_norm=True),

            ConvTransposeLayer(in_channels=hid_channels * 2,
                               out_channels=hid_channels,
                               batch_norm=True),

            torch.nn.ConvTranspose2d(in_channels=hid_channels,
                                     out_channels=out_channels,
                                     kernel_size=4,
                                     stride=2,  # from architecture guidelines in DCGAN paper
                                     padding=1,
                                     bias=False),
            torch.nn.Tanh()  # from architecture guidelines in DCGAN paper
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of DCGAN Discriminator"""
        generated_data = self.model(x)
        return generated_data
