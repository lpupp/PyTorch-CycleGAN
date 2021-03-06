"""Calculate GAN evaluation metrics.

Source:
Xu, Q., Huang, G., Yuan, Y., Guo, C., Sun, Y., Wu, F., & Weinberger, K. (2018).
An empirical study on evaluation metrics of generative adversarial networks.
arXiv preprint arXiv:1806.07755.

https://github.com/xuqiantong/GAN-Metrics/blob/master/metric.py

TODO This is an out-of-sample calculate-metrics script. duplicate to in-sample...
"""

import os
import argparse

import ot
import math
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data

from scipy import linalg

from models import Generator
from datasets import ImageDataset

# class MakeDataset(torch.utils.data.Dataset):
#     """Load images from my folder structure."""
#
#     def __init__(self, imgs, loader, transform=None):
#         self.samples = imgs
#         self.transform = transform
#         self.loader = loader
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, index):
#         path = self.samples[index]
#         sample = self.loader(path)
#         if self.transform is not None:
#             sample = self.transform(sample)
#
#         return sample


def to_torch(x):
    return dict((k, torch.cat(v, 0).to('cpu')) for k, v in x.items())


class ConvNetFeatureSaver(object):
    def __init__(self, model='resnet34'):
        """Init.

        model: inception_v3, vgg13, vgg16, vgg19, resnet18, resnet34,
               resnet50, resnet101, or resnet152
        """
        self.model = model

        self.names = ['A', 'B', 'BA', 'AB']
        self.feature_conv = {key: [] for key in self.names}
        self.feature_logit = {key: [] for key in self.names}
        self.feature_smax = {key: [] for key in self.names}

        if self.model.find('vgg') >= 0:
            self.vgg = getattr(models, model)(pretrained=True).cuda().eval()
            self.trans = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        elif self.model.find('resnet') >= 0:
            resnet = getattr(models, model)(pretrained=True)
            resnet.cuda().eval()
            resnet_feature = nn.Sequential(resnet.conv1, resnet.bn1,
                                           resnet.relu,
                                           resnet.maxpool, resnet.layer1,
                                           resnet.layer2, resnet.layer3,
                                           resnet.layer4).cuda().eval()
            self.resnet = resnet
            self.resnet_feature = resnet_feature
            self.trans = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        elif self.model == 'inception' or self.model == 'inception_v3':
            inception = models.inception_v3(
                pretrained=True, transform_input=False).cuda().eval()
            inception_feature = nn.Sequential(inception.Conv2d_1a_3x3,
                                              inception.Conv2d_2a_3x3,
                                              inception.Conv2d_2b_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Conv2d_3b_1x1,
                                              inception.Conv2d_4a_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Mixed_5b,
                                              inception.Mixed_5c,
                                              inception.Mixed_5d,
                                              inception.Mixed_6a,
                                              inception.Mixed_6b,
                                              inception.Mixed_6c,
                                              inception.Mixed_6d,
                                              inception.Mixed_7a,
                                              inception.Mixed_7b,
                                              inception.Mixed_7c,
                                              ).cuda().eval()
            self.inception = inception
            self.inception_feature = inception_feature
            self.trans = transforms.Compose([
                transforms.Resize(299),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            raise NotImplementedError

    def save(self, dataloader, G_A2B, G_B2A):  # real_A, real_B, fake_BA, fake_AB):
        print('extracting features...')
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if args.cuda else torch.Tensor
        input_A = Tensor(args.batch_size, 3, 3, args.img_size)
        input_B = Tensor(args.batch_size, 3, 3, args.img_size)

        for j, batch in enumerate(dataloader):
            if j <= args.sample_size:
                # Set model input
                real_A = Variable(input_A.copy_(batch['A']))
                real_B = Variable(input_B.copy_(batch['B']))

                # Generate output
                fake_AB = 0.5*(netG_A2B(real_A).data + 1.0)
                fake_BA = 0.5*(netG_B2A(real_B).data + 1.0)

                for i, input in zip(self.names, [real_A, real_B, fake_BA, fake_AB]):
                    input_ = input.unsqueeze(0) if input.dim() == 3 else input
                    with torch.no_grad():
                        if self.model == 'vgg' or self.model == 'vgg16':
                            fconv = self.vgg.features(input_).view(input.size(0), -1)
                            flogit = self.vgg.classifier(fconv)
                            # flogit = self.vgg.logitifier(fconv)
                        elif self.model.find('resnet') >= 0:
                            fconv = self.resnet_feature(
                                input_).mean(3).mean(2).squeeze()
                            flogit = self.resnet.fc(fconv)
                        elif self.model == 'inception' or self.model == 'inception_v3':
                            fconv = self.inception_feature(
                                input_).mean(3).mean(2).squeeze()
                            flogit = self.inception.fc(fconv)
                        else:
                            raise NotImplementedError

                        fsmax = F.softmax(flogit, dim=0)
                        #self.feature_pixl.append(img)
                        self.feature_conv[i].append(fconv.data.cpu())
                        self.feature_logit[i].append(flogit.data.cpu())
                        self.feature_smax[i].append(fsmax.data.cpu())
            else:
                break

    def calculate_scores(self):
        #feature_pixl = torch.cat(self.feature_pixl, 0).to('cpu')
        self.feature_conv = to_torch(self.feature_conv)
        self.feature_logit = to_torch(self.feature_logit)
        self.feature_smax = to_torch(self.feature_smax)

        out = []
        for i in 'AB':
            score = []
            j = 'BA' if i == 'A' else 'AB'

            for nm, conv in zip(['conv', 'logit', 'smax'],
                                [self.feature_conv, self.feature_logit, self.feature_smax]):

                Mxx = distance(conv[i], conv[i], False)
                Mxy = distance(conv[i], conv[j], False)
                Myy = distance(conv[j], conv[j], False)

                tmp = knn(Mxx, Mxy, Myy, 1, False)
                score += [tmp.acc, tmp.acc_t, tmp.acc_f, tmp.precision, tmp.recall]

            score.append(mode_score(self.feature_smax[i], self.feature_smax[j]))

            out.append(score)

        self.re_init()

        return out

    def re_init(self):
        self.feature_conv = {key: [] for key in self.names}
        self.feature_logit = {key: [] for key in self.names}
        self.feature_smax = {key: [] for key in self.names}


def distance(X, Y, sqrt):
    """Calculate distance."""
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX, -1)
    X2 = (X*X).sum(1).resize_(nX, 1)
    Y = Y.view(nY, -1)
    Y2 = (Y*Y).sum(1).resize_(nY, 1)

    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) -
            2 * torch.mm(X, Y.transpose(0, 1)))

    del X, X2, Y, Y2

    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()

    return M


def wasserstein(M, sqrt):
    """Calculate earth mover's distance."""
    if sqrt:
        M = M.abs().sqrt()
    emd = ot.emd2([], [], M.numpy())

    return emd


class ScoreKNN:
    """Store KNN scores."""

    acc = 0
    acc_real = 0
    acc_fake = 0
    precision = 0
    recall = 0
    tp = 0
    fp = 0
    fn = 0
    ft = 0


def knn(Mxx, Mxy, Myy, k, sqrt):
    """Calculate KNN."""
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1)))
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat(
        (Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1))
                ).topk(k, 0, False)

    count = torch.zeros(n0 + n1)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1)).float()

    s = ScoreKNN()
    s.tp = (pred * label).sum()
    s.fp = (pred * (1 - label)).sum()
    s.fn = ((1 - pred) * label).sum()
    s.tn = ((1 - pred) * (1 - label)).sum()
    s.precision = s.tp / (s.tp + s.fp + 1e-10)
    s.recall = s.tp / (s.tp + s.fn + 1e-10)
    s.acc_t = s.tp / (s.tp + s.fn)
    s.acc_f = s.tn / (s.tn + s.fp)
    s.acc = torch.eq(label, pred).float().mean()
    s.k = k

    return s


def mmd(Mxx, Mxy, Myy, sigma):
    """Calculate maximum mean discrepancy."""
    scale = Mxx.mean()
    Mxx = torch.exp(-Mxx / (scale * 2 * sigma * sigma))
    Mxy = torch.exp(-Mxy / (scale * 2 * sigma * sigma))
    Myy = torch.exp(-Myy / (scale * 2 * sigma * sigma))
    mmd = math.sqrt(Mxx.mean() + Myy.mean() - 2 * Mxy.mean())

    return mmd


eps = 1e-20


def inception_score(X):
    """Calculate inception score."""
    kl = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
    score = np.exp(kl.sum(1).mean())

    return score


def mode_score(X, Y):
    """Calculate mode score."""
    kl1 = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
    kl2 = X.mean(0) * ((X.mean(0)+eps).log()-(Y.mean(0)+eps).log())
    score = np.exp(kl1.sum(1).mean() - kl2.sum())

    return score


def fid(X, Y):
    """Calculate Frechet inception score."""
    m = X.mean(0)
    m_w = Y.mean(0)
    X_np = X.numpy()
    Y_np = Y.numpy()

    C = np.cov(X_np.transpose())
    C_w = np.cov(Y_np.transpose())
    C_C_w_sqrt = linalg.sqrtm(C.dot(C_w), True).real

    score = m.dot(m) + m_w.dot(m_w) - 2 * m_w.dot(m) + \
        np.trace(C + C_w - 2 * C_C_w_sqrt)
    return np.sqrt(score)


def compute_score_raw(feature_r, feature_f):
    """Compute GAN evaluation scores."""
    # 4 feature spaces and 7 scores + incep + modescore + fid
    score = np.zeros(4 * 7 + 3)
    for i in range(0, 4):
        print('compute score in space: ' + str(i))
        Mxx = distance(feature_r[i], feature_r[i], False)
        Mxy = distance(feature_r[i], feature_f[i], False)
        Myy = distance(feature_f[i], feature_f[i], False)

        score[i * 7] = wasserstein(Mxy, True)
        score[i * 7 + 1] = mmd(Mxx, Mxy, Myy, 1)
        tmp = knn(Mxx, Mxy, Myy, 1, False)
        score[(i * 7 + 2):(i * 7 + 7)] = \
            tmp.acc, tmp.acc_t, tmp.acc_f, tmp.precision, tmp.recall

    score[28] = inception_score(feature_f[3])
    score[29] = mode_score(feature_r[3], feature_f[3])
    score[30] = fid(feature_r[3], feature_f[3])
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate GAN metrics.')

    parser.add_argument('--seed', type=int, help='manual seed')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)

    parser.add_argument('--mode', type=str, default='None', help='train|val.')
    parser.add_argument('--n_iter', type=int, default=25, help='Iter number to calculate metrics on.')

    parser.add_argument('--models_path', type=str, default='None', help='Path to models.')
    parser.add_argument('--img_path', type=str, default='/home/cluster/lgaega/data/fashion/shoes/img', help='Path to training images (A).')

    parser.add_argument('--output_path', type=str, default='/home/cluster/lgaega/perfectpairings/output/DiscoGAN/GAN_metrics/fashion/shoes2dresses/DiscoGAN_extra64', help='Set the path the result images will be saved.')

    parser.add_argument('--batch_size', type=str, default=64, help='batch size')
    parser.add_argument('--sample_size', type=int, default=None, help='number of samples for evaluation')
    parser.add_argument('--img_size', type=int, default=256, help='size of the data (squared assumed)')

    parser.add_argument('--conv_model', type=str, default='inception_v3',
                        help='inception_v3|vgg13|vgg16|vgg19|resnet18|resnet34|resnet50|resnet101|resnet152')

    global args
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_path):
        try:
            os.makedirs(args.output_path)
        except OSError:
            pass

    seed = args.seed or random.randint(1, 10000)
    print('Seed:', seed)
    random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = True

    # load model
    model_path_A2B = os.path.join(args.models_path, str(float(args.n_iter)))
    model_path_A2B = os.path.join(args.models_path, str(float(args.n_iter)))

    netG_A2B = Generator(3, 3)
    netG_B2A = Generator(3, 3)

    if args.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(model_path_A2B))
    netG_B2A.load_state_dict(torch.load(model_path_A2B))

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()

    # Dataset loader
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    dataloader = DataLoader(ImageDataset(args.img_path, transforms_=transforms_, mode=args.mode),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    convnet_feature_saver = ConvNetFeatureSaver(model=args.conv_model,
                                                batch_size=args.batch_size,
                                                workers=args.workers)

    convnet_feature_saver.save(dataloader, netG_A2B, netG_B2A)
    scores = convnet_feature_saver.calculate_scores()

    scores.append((scores[0] + scores[1]) / 2)

    # Save to csv
    csv_fn = os.path.join(args.output_path, 'score_{}.csv')
    with open(csv_fn.format('A'), 'a') as f:
        f.write('\n' + ','.join(str(e) for e in scores[0]))

    with open(csv_fn.format('B'), 'a') as f:
        f.write('\n' + ','.join(str(e) for e in scores[1]))

    with open(csv_fn.format('mean'), 'a') as f:
        f.write('\n' + ','.join(str(e) for e in scores[2]))

    # Save to numpy
    np.save(os.path.join(args.output_path, 'score_A_{}.npy'.format(args.n_iter)), np.array(scores[0]))
    np.save(os.path.join(args.output_path, 'score_B_{}.npy'.format(args.n_iter)), np.array(scores[1]))
    np.save(os.path.join(args.output_path, 'score_mean_{}.npy'.format(args.n_iter)), np.array(scores[2]))
    print('Training completed')
    print('Metric scores output to {}'.format(args.output_path))
