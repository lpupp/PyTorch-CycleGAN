"""Mini-batch discrimination pytorch implementation.

source:
https://gist.github.com/t-ae/732f78671643de97bbe2c46519972491
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal_(self.T)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x


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
                 img_size=64, keep_weights_proportional=False,
                 extra_layer=False, upsample=False):
        super(Generator, self).__init__()

        if keep_weights_proportional:
            if upsample:
                if img_size == 64:
                    filter_dim, n_padding = 3, 1
                elif img_size == 128:
                    filter_dim, n_padding = 7, 3
                elif img_size == 256:
                    filter_dim, n_padding = 13, 6
                else:
                    raise NotImplementedError
            else:
                raise ValueError('Cannot keep_weights_proportional if upsample False')
        else:
            filter_dim, n_padding = 3, 1

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
                model += [nn.Conv2d(in_features, in_features, filter_dim, padding=n_padding),
                          nn.InstanceNorm2d(in_features),
                          nn.ReLU(inplace=True)]

            model += [nn.Conv2d(in_features, out_features, filter_dim, stride=2, padding=n_padding),
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
                    model += [nn.Conv2d(in_features, in_features, filter_dim, padding=n_padding),
                              nn.InstanceNorm2d(in_features),
                              nn.ReLU(inplace=True)]

                model += [nn.Conv2d(in_features, out_features, filter_dim, padding=n_padding),
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

                model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
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
    def __init__(self, input_nc, extra_layer=False, mb_D=False, x_size=None):
        super(Discriminator, self).__init__()

        # # A bunch of convolutions one after another
        # model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
        #          nn.LeakyReLU(0.2, inplace=True)]
        #
        # model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
        #           nn.InstanceNorm2d(128),
        #           nn.LeakyReLU(0.2, inplace=True)]
        #
        # if extra_layer:
        #     model += [nn.Conv2d(128, 128, 3, padding=1),
        #               nn.InstanceNorm2d(128),
        #               nn.LeakyReLU(0.2, inplace=True)]
        #
        # model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
        #           nn.InstanceNorm2d(256),
        #           nn.LeakyReLU(0.2, inplace=True)]
        #
        # if extra_layer:
        #     model += [nn.Conv2d(256, 256, 3, padding=1),
        #               nn.InstanceNorm2d(256),
        #               nn.LeakyReLU(0.2, inplace=True)]
        #
        # model += [nn.Conv2d(256, 512, 4, padding=1),
        #           nn.InstanceNorm2d(512),
        #           nn.LeakyReLU(0.2, inplace=True)]
        #
        # # FCN classification layer
        # model += [nn.Conv2d(512, 1, 4, padding=1)]
        #
        # self.model = nn.Sequential(*model)

        self.extra_layer = extra_layer
        self.mb_D = mb_D
        if self.mb_D:
            assert x_size is not None, 'provide x_dim when mb_D True'

        # A bunch of convolutions one after another
        self.conv1 = nn.Conv2d(input_nc, 64, 4, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        if self.extra_layer:
            self.conv2e = nn.Conv2d(128, 128, 3, padding=1)
            self.in2e = nn.InstanceNorm2d(128)
            self.relu2e = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        if self.extra_layer:
            self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
            self.in3 = nn.InstanceNorm2d(256)
            self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(256, 512, 4, padding=1)
        self.in4 = nn.InstanceNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        if self.mb_D:
            flat_dim = ((int(x_size/8)-1)**2) * 512
            self.mb = MinibatchDiscrimination(flat_dim, 512, 64)
            self.dense = nn.Linear(512 + flat_dim, 1)
        else:
            self.conv5 = nn.Conv2d(512, 1, 4, padding=1)

    def forward(self, x):
        # x = self.model(x)
        # # Average pooling and flatten
        # return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)  # return flattened avg_pool2d

        if self.extra_layer:
            conv1 = self.conv1(x)
            relu1 = self.relu1(conv1)

            conv2 = self.conv2(relu1)
            in2 = self.in2(conv2)
            relu2 = self.relu2(in2)

            conv2e = self.conv2e(relu2)
            in2e = self.in2e(conv2e)
            relu2e = self.relu2e(in2e)

            conv3 = self.conv3(relu2e)
            in3 = self.in3(conv3)
            relu3 = self.relu3(in3)

            conv3e = self.conv3e(relu3)
            in3e = self.in3e(conv3e)
            relu3e = self.relu3e(in3e)

            conv4 = self.conv4(relu3e)
            in4 = self.in4(conv4)
            relu4 = self.relu4(in4)

            relus = [relu2, relu2e, relu3, relu3e, relu4]

        else:
            conv1 = self.conv1(x)
            relu1 = self.relu1(conv1)

            conv2 = self.conv2(relu1)
            in2 = self.in2(conv2)
            relu2 = self.relu2(in2)

            conv3 = self.conv3(relu2)
            in3 = self.in3(conv3)
            relu3 = self.relu3(in3)

            conv4 = self.conv4(relu3)
            in4 = self.in4(conv4)
            relu4 = self.relu4(in4)

            relus = [relu2, relu3, relu4]

        if self.mb_D:
            relu4_flat = relu4.view(relu4.size(0), -1)
            mb = self.mb(relu4_flat)
            dense = self.dense(mb)
            out = torch.sigmoid(dense)
        else:
            conv5 = self.conv5(relu4)
            out = F.avg_pool2d(conv5, conv5.size()[2:]).view(conv5.size(0), -1)

        return out, relus
