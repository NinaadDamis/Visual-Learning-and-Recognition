import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F


class UpSampleConv2D(jit.ScriptModule):
    # TODO 1.1: Implement nearest neighbor upsampling + conv layer

    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        self.in_channels    = input_channels
        self.kernel_size    = kernel_size
        self.n_filters      = n_filters
        self.upscale_factor = upscale_factor
        self.padding        = padding
        self.pixelshuffle   = nn.PixelShuffle(upscale_factor)
        self.conv           = nn.Conv2d(self.in_channels,self.n_filters,self.kernel_size,padding=self.padding)

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement nearest neighbor upsampling.
        # 1. Duplicate x channel wise upscale_factor^2 times.
        # 2. Then re-arrange to form an image of shape (batch x channel x height*upscale_factor x width*upscale_factor).
        # 3. Apply convolution.
        # Hint for 2. look at
        # https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle
        duplicate_x = x.repeat(1,self.upscale_factor*self.upscale_factor,1,1)
        out         = self.pixelshuffle(duplicate_x)
        out         = self.conv(out)
        return out

class DownSampleConv2D(jit.ScriptModule):
    # TODO 1.1: Implement spatial mean pooling + conv layer

    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        downscale_ratio=2,
        padding=0,
        stride=1
    ):
        super(DownSampleConv2D, self).__init__()
        self.in_channels     = input_channels
        self.kernel_size     = kernel_size
        self.n_filters       = n_filters
        self.downscale_ratio = downscale_ratio
        self.padding         = padding
        self.pixelunshuffle  = nn.PixelUnshuffle(self.downscale_ratio)
        self.conv            = nn.Conv2d(self.in_channels,self.n_filters,self.kernel_size,padding=self.padding)
        
    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement spatial mean pooling
        # 1. Re-arrange to form a batch x channel * upscale_factor^2 x height x width
        # 2. Then split channel wise into batch x channel x height x width Images
        # 3. average the images into one and apply convolution
        # Hint for 1. look at
        # https://pytorch.org/docs/master/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle

        out = self.pixelunshuffle(x)
        # Use chunk
        # https://pytorch.org/docs/stable/generated/torch.chunk.html
        out = out.chunk(int(self.downscale_ratio * self.downscale_ratio), dim = 1)
        mu  = torch.mean(torch.stack(list(out)),dim = 0)
        sqz = torch.squeeze(mu)
        ret = self.conv(sqz)
        return ret


class ResBlockUp(jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Upsampler.
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(in_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.layers = nn.Sequential(
            nn.BatchNorm2d(self.input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(self.input_channels, self.n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.residual = UpSampleConv2D(self.n_filters,self.kernel_size,self.n_filters,padding=1)
        self.shortcut = UpSampleConv2D(self.input_channels,1,self.n_filters)

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
        layer_out = self.layers(x)
        residual_out = self.residual(layer_out)
        shortcut_out = self.shortcut(x)
        residual_out += shortcut_out
        # print("Return ResBlockUp() forward")
        return residual_out

class ResBlockDown(jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Downsampler.
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
        )
        (residual): DownSampleConv2D(
            (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): DownSampleConv2D(
            (conv): Conv2d(in_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        # print("Initialize ResblockDown()")
        self.input_channels = input_channels
        self.kernel_size    = kernel_size
        self.n_filters      = n_filters
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.input_channels, self.n_filters, self.kernel_size, stride=(1, 1), padding=(1, 1)),
            nn.ReLU() 
        )
        self.residual = DownSampleConv2D(self.n_filters,self.kernel_size,self.n_filters,padding=1)
        self.shortcut = DownSampleConv2D(self.input_channels,1,self.n_filters)

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
        # print("Entered ResBlockDOwn Forward()")
        layers_out = self.layers(x)
        residual_out = self.residual(layers_out)
        shortcut_out = self.shortcut(x)
        residual_out +=shortcut_out
        return residual_out


class ResBlock(jit.ScriptModule):
    # TODO 1.1: Impement Residual Block as described below.
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        # print("Initialize Resblock()")
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=(1, 1))
        )

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the conv layers. Don't forget the residual connection!
        layers_out = self.layers(x)
        # print("ResBlock() : Shapes layers_out and x (should be same )", layers_out.shape,x.shape)
        return layers_out + x


class Generator(jit.ScriptModule):
    # TODO 1.1: Impement Generator. Follow the architecture described below:
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
            (layers): Sequential(
                (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (1): ReLU()
                (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
            (layers): Sequential(
                (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (1): ReLU()
                (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        # print("Initialized Generator")
        self.dense  = nn.Linear(in_features=128, out_features=2048, bias=True)
        self.layers = nn.Sequential(

            ResBlockUp(input_channels=128,kernel_size=3,n_filters=128),
            ResBlockUp(input_channels=128,kernel_size=3,n_filters=128),
            ResBlockUp(input_channels=128,kernel_size=3,n_filters=128),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Tanh()
        )

    @jit.script_method
    def forward_given_samples(self, z):
        # TODO 1.1: forward the generator assuming a set of samples z have been passed in.
        # Don't forget to re-shape the output of the dense layer into an image with the appropriate size!
        out = self.dense(z)
        # Input to layers,ie,resblock, requires 128 channels. Out of dense is 2048.
        #2048/ 128 = 16 . h *w = 16 -> h=w=4. Starting image size given lol.
        out = out.view(-1,128,4,4)
        # print("Forward given samples() :Reshape after dense = ", out.shape)
        out = self.layers(out)

        return out

    @jit.script_method
    def forward(self, n_samples: int = 1024):
        # TODO 1.1: Generate n_samples latents and forward through the network.
        # Make sure to cast the latents to type half (for compatibility with torch.cuda.amp.autocast)


        #https://pytorch.org/docs/stable/generated/torch.Tensor.half.html
        # use .half() or .to(float16)
        #n_samples is basically a batch size param.
        latents = torch.normal(0.0,1.0, (n_samples,128))
        # print("Generator() : Forward() - Latents shape = ", latents.shape)
        latents = latents.half().cuda()
        out = self.dense(latents)
        out = out.view(-1,128,4,4)
        out = self.layers(out)
        # print("Return Gen forward()")
        return out
        


class Discriminator(jit.ScriptModule):
    # TODO 1.1: Impement Discriminator. Follow the architecture described below:
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
            )
            (residual): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (shortcut): DownSampleConv2D(
                (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
            )
        )
        (1): ResBlockDown(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
            )
            (residual): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (shortcut): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            )
        )
        (2): ResBlock(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (3): ResBlock(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        # print("Initialized Discriminator")
        self.dense  = nn.Linear(in_features=128, out_features=1, bias=True)
        self.layers = nn.Sequential(
            ResBlockDown(input_channels=3,kernel_size=3,n_filters=128),
            ResBlockDown(input_channels=128,kernel_size=3,n_filters=128),
            ResBlock(input_channels=128,kernel_size=3,n_filters=128),
            ResBlock(input_channels=128,kernel_size=3,n_filters=128),
            nn.ReLU()
        )
        self.resb = ResBlockDown(input_channels=3,kernel_size=3,n_filters=128)

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the discriminator assuming a batch of images have been passed in.
        # Make sure to flatten the output of the convolutional layers and sum across the image dimensions before passing to the output layer!
        # print(" discrim forward")
        aa = self.resb(x)
        # print("SOmething work")
        out = self.layers(x)
        # print("Out of layers forward")
        # Output of resblock is batch,128,x,x --> I guess sum across these two as input to dense is 128 feats.
        out = torch.sum(out,(2,3))
        # print("Discriminator forward : Shape of out after summing = (should be (batch,128))", out.shape)
        out = self.dense(out)

        return out
