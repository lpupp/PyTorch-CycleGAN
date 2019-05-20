import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Interpolate(nn.Module):
    def __init__(self, scale_factor, size=None, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.size = size
        self.scale = scale_factor
        self.mode = mode

    def forward(self, x):
        #x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        x = self.interp(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9,
                 #filter_dim=3,
                 extra_layer=False, upsample=False):
        super(Generator, self).__init__()

        filter_dim = 3
        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features*2

        # W=(W−F+2P)/S+1
        for _ in range(2):
            if extra_layer:
                model += [nn.Conv2d(in_features, in_features, 3, padding=1),
                          nn.InstanceNorm2d(in_features),
                          nn.ReLU(inplace=True)]

            model += [nn.Conv2d(in_features, out_features, filter_dim, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        if upsample:
            for _ in range(2):
                if extra_layer:
                    model += [nn.Conv2d(in_features, in_features, 3, padding=1),
                              nn.InstanceNorm2d(in_features),
                              nn.ReLU(inplace=True)]

                model += [nn.Conv2d(in_features, out_features, 3, padding=1),
                          nn.InstanceNorm2d(out_features),
                          nn.ReLU(inplace=True),
                          Interpolate(2.0)]

                in_features = out_features
                out_features = in_features//2

            # Output layer
            model += [nn.ReflectionPad2d(3),
                      nn.Conv2d(64, output_nc, 7),
                      nn.Tanh()]

            self.model = nn.Sequential(*model)

        else:
            for _ in range(2):
                if extra_layer:
                    model += [nn.ConvTranspose2d(in_features, in_features, 3, padding=1),
                              nn.InstanceNorm2d(in_features),
                              nn.ReLU(inplace=True)]

                model += [nn.ConvTranspose2d(in_features, out_features, filter_dim, stride=2, padding=1, output_padding=1),
                          nn.InstanceNorm2d(out_features),
                          nn.ReLU(inplace=True)]
                in_features = out_features
                out_features = in_features//2

            # Output layer
            model += [nn.ReflectionPad2d(3),
                      nn.Conv2d(64, output_nc, 7),
                      nn.Tanh()]

            self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc, extra_layer=False):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        if extra_layer:
            model += [nn.Conv2d(128, 128, 3, padding=1),
                      nn.InstanceNorm2d(128),
                      nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        if extra_layer:
            model += [nn.Conv2d(256, 256, 3, padding=1),
                      nn.InstanceNorm2d(256),
                      nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
