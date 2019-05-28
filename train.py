"""CycleGAN implementation Pytorch.

mode collapse techniques:
- mini-batch discrimination (implemented as --mb_D)
- wasserstein loss (implemented as --wasserstein)
- experience replace (implemeted as ReplayBuffer: dim = 50)
https://medium.com/@utk.is.here/training-a-conditional-dc-gan-on-cifar-10-fce88395d610

- decay cycle (reconstruction) loss (implemented as --recon_loss_acay and --recon_acay_rate <- [0.0 1.0) )
https://ssnl.github.io/better_cycles/report.pdf
"""
import os
import argparse
import itertools

import imageio

import numpy as np
import time

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator, Discriminator
from utils import ReplayBuffer, LambdaLR, weights_init_normal  # , Logger
from datasets import ImageDataset

#from eval_score import ConvNetFeatureSaver

parser = argparse.ArgumentParser()
parser.add_argument('--load_iter', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--n_test', type=int, default=None, help='number of test samples to spit out')
parser.add_argument('--n_sample', type=int, default=60, help='number ofsample samples to spit out')

parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='data/fashion/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

parser.add_argument('--buffer_size', type=int, default=50, help='Size of replay buffer')

parser.add_argument('--upsample', action='store_true', help='If True: upsample; else: transposed 2D conv')
parser.add_argument('--keep_prop', action='store_true', help='Keep weights proportional to 3:64 ratio')
parser.add_argument('--G_extra', action='store_true', help='use extra layers in G')
parser.add_argument('--D_extra', action='store_true', help='use extra layers in D')
parser.add_argument('--slow_D', action='store_true', help='Slow the training of D to avoid mode collapse')
parser.add_argument('--wasserstein', action='store_true', help='use wasserstein loss for D')
parser.add_argument('--mb_D', action='store_true', help='Mini-batch discrimination')
parser.add_argument('--fm_loss', action='store_true', help='Feature matching loss')
parser.add_argument('--add_noise', action='store_true', help='Add noise to training images when loading')

parser.add_argument('--recon_loss_epoch', type=int, default=100, help='epoch to start linearly adapting recon loss weight')
parser.add_argument('--start_recon_loss_val', type=float, default=10.0, help='starting weight of recon loss (cycleGAN default = 10.0)')
parser.add_argument('--end_recon_loss_val', type=float, default=10.0, help='end weight of recon loss (cycleGAN default = 10.0)')

parser.add_argument('--gan_loss_epoch', type=int, default=100, help='epoch to start linearly adapting recon loss weight')
parser.add_argument('--start_gan_loss_val', type=float, default=1.0, help='starting weight of gan loss (cycleGAN default = 1.0)')
parser.add_argument('--end_gan_loss_val', type=float, default=1.0, help='end weight of gan loss (cycleGAN default = 1.0)')

parser.add_argument('--img_norm', type=str, default='znorm', help='How to normalize images: znorm|scale01|scale01flip')

parser.add_argument('--horizontal_flip', action='store_true', help='augment data by flipping horizontally')
parser.add_argument('--resize_crop', action='store_true', help='augment reading in image too large and cropping')

parser.add_argument('--output_dir', type=str, default='data/output/fashion/shoes2dresses', help='output directory')
parser.add_argument('--score_interval', type=int, default=50, help='Calculate out-of-sample scores every score_interval iterations.')
parser.add_argument('--log_interval', type=int, default=50, help='Print loss values every log_interval iterations.')
parser.add_argument('--plot_interval', type=int, default=50, help='Print loss values every plot_interval iterations.')
parser.add_argument('--image_save_interval', type=int, default=1000, help='Save test results every image_save_interval iterations.')
parser.add_argument('--model_save_interval', type=int, default=1000, help='Save models every model_save_interval iterations.')


def safe_mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def tensor2image(tensor, img_norm='znorm'):
    image = as_np(tensor)
    if img_norm == 'znorm':
        image = 127.5*(image + 1.0)
    elif 'scale01' in img_norm:
        if img_norm == 'scale01flip':
            image = np.absolute(image - 1.0)
        image = image * 255
    else:
        raise NotImplementedError

    return image.astype(np.uint8).transpose(1, 2, 0)


def as_np(tensor):
    return tensor.cpu().float().detach().numpy()


def get_fm_loss(real_feats, fake_feats, criterion, cuda):
    losses = 0
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        if cuda:
            loss = criterion(l2, Variable(torch.ones(l2.size())).cuda())
        else:
            loss = criterion(l2, Variable(torch.ones(l2.size())))
        losses += loss

    return losses


def wasserstein_loss(prediction, target_is_real):
    if target_is_real:
        loss = - prediction.mean()
    else:
        loss = prediction.mean()
    return loss


def main(args):
    torch.manual_seed(0)
    if args.mb_D:
        raise NotImplementedError('mb_D not implemented')
        assert args.batch_size > 1, 'batch size needs to be larger than 1 if mb_D'

    if args.img_norm != 'znorm':
        raise NotImplementedError('{} not implemented'.format(args.img_norm))

    modelarch = 'C_{0}_{1}_{2}{3}{4}{5}{6}{7}{8}{9}{10}{11}{12}{13}{14}{15}'.format(
        args.size, args.batch_size, args.lr,  # 0, 1, 2
        '_G' if args.G_extra else '',  # 3
        '_D' if args.D_extra else '',  # 4
        '_U' if args.upsample else '',  # 5
        '_S' if args.slow_D else '',  # 6
        '_RL{}-{}'.format(args.start_recon_loss_val, args.start_recon_loss_val),  # 7
        '_GL{}-{}'.format(args.start_gan_loss_val, args.start_gan_loss_val),  # 8
        '_prop' if args.keep_prop else '',  # 9
        '_' + args.img_norm,  # 10
        '_WL' if args.wasserstein else '',  # 11
        '_MBD' if args.mb_D else '',  # 12
        '_FM' if args.fm_loss else '',  # 13
        '_BF{}'.format(args.buffer_size) if args.buffer_size != 50 else '',  # 14
        '_N' if args.add_noise else '')  # 15

    samples_path = os.path.join(args.output_dir, modelarch, 'samples')
    safe_mkdirs(samples_path)
    model_path = os.path.join(args.output_dir, modelarch, 'models')
    safe_mkdirs(model_path)
    test_path = os.path.join(args.output_dir, modelarch, 'test')
    safe_mkdirs(test_path)

    # Definition of variables ######
    # Networks
    netG_A2B = Generator(args.input_nc, args.output_nc, img_size=args.size, extra_layer=args.G_extra, upsample=args.upsample, keep_weights_proportional=args.keep_prop)
    netG_B2A = Generator(args.output_nc, args.input_nc, img_size=args.size, extra_layer=args.G_extra, upsample=args.upsample, keep_weights_proportional=args.keep_prop)
    netD_A = Discriminator(args.input_nc, extra_layer=args.D_extra, mb_D=args.mb_D, x_size=args.size)
    netD_B = Discriminator(args.output_nc, extra_layer=args.D_extra, mb_D=args.mb_D, x_size=args.size)

    if args.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # Lossess
    criterion_GAN = wasserstein_loss if args.wasserstein else torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    feat_criterion = torch.nn.HingeEmbeddingLoss()

    # I could also update D only if iters % 2 == 0
    lr_G = args.lr
    lr_D = args.lr / 2 if args.slow_D else args.lr

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr_G, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr_D, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.n_epochs, args.load_iter, args.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(args.n_epochs, args.load_iter, args.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(args.n_epochs, args.load_iter, args.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.Tensor
    input_A = Tensor(args.batch_size, args.input_nc, args.size, args.size)
    input_B = Tensor(args.batch_size, args.output_nc, args.size, args.size)
    target_real = Variable(Tensor(args.batch_size).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(args.batch_size).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer(args.buffer_size)
    fake_B_buffer = ReplayBuffer(args.buffer_size)

    # Transforms and dataloader for training set
    transforms_ = []
    if args.resize_crop:
        transforms_ += [transforms.Resize(int(args.size*1.12), Image.BICUBIC),
                        transforms.RandomCrop(args.size)]
    else:
        transforms_ += [transforms.Resize(args.size, Image.BICUBIC)]

    if args.horizontal_flip:
        transforms_ += [transforms.RandomHorizontalFlip()]

    transforms_ += [transforms.ToTensor()]

    if args.add_noise:
        transforms_ += [transforms.Lambda(lambda x: x + torch.randn_like(x))]

    transforms_norm = []
    if args.img_norm == 'znorm':
        transforms_norm += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    elif 'scale01' in args.img_norm:
        transforms_norm += [transforms.Lambda(lambda x: x.mul(1/255))]
        if 'flip' in args.img_norm:
            transforms_norm += [transforms.Lambda(lambda x: (x - 1).abs())]
    else:
        raise ValueError('wrong --img_norm. only znorm|scale01|scale01flip')

    transforms_ += transforms_norm

    dataloader = DataLoader(ImageDataset(args.dataroot, transforms_=transforms_, unaligned=True),
                            batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)

    # Transforms and dataloader for test set
    transforms_test_ = [transforms.Resize(args.size, Image.BICUBIC),
                        transforms.ToTensor()]
    transforms_test_ += transforms_norm

    dataloader_test = DataLoader(ImageDataset(args.dataroot, transforms_=transforms_test_, mode='test'),
                                 batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)
    # Training ######
    iter = 0
    prev_time = time.time()
    n_test = 10e10 if args.n_test is None else args.n_test
    n_sample = 10e10 if args.n_sample is None else args.n_sample

    #gan_metrics = ConvNetFeatureSaver()
    #csv_fn = os.path.join(args.output_dir, modelarch, modelarch + '.csv')

    rl_delta_x = args.n_epochs - args.recon_loss_epoch
    rl_delta_y = args.end_recon_loss_val - args.start_recon_loss_val

    gan_delta_x = args.n_epochs - args.gan_loss_epoch
    gan_delta_y = args.end_gan_loss_val - args.start_gan_loss_val

    for epoch in range(args.load_iter, args.n_epochs):

        rl_effective_epoch = max(epoch - args.recon_loss_epoch, 0)
        recon_loss_rate = args.start_recon_loss_rate + rl_effective_epoch * (rl_delta_y / rl_delta_x)

        gan_effective_epoch = max(epoch - args.gan_loss_epoch, 0)
        gan_loss_rate = args.start_recon_loss_rate + gan_effective_epoch * (gan_delta_y / gan_delta_x)

        id_loss_rate = 5.0

        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            # Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake, _ = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake, _ = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)

            # Total loss
            loss_G = (loss_identity_A + loss_identity_B) * id_loss_rate
            loss_G += (loss_GAN_A2B + loss_GAN_B2A) * gan_loss_rate
            loss_G += (loss_cycle_ABA + loss_cycle_BAB) * recon_loss_rate

            loss_G.backward()

            optimizer_G.step()

            # Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real, _ = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake, _ = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

            if args.fm_loss:
                pred_real, feats_real = netD_A(real_A)
                pred_fake, feats_fake = netD_A(fake_A.detach())

                fm_loss_A = get_fm_loss(feats_real, feats_fake, feat_criterion, args.cuda)

                loss_D_A = loss_D_A * 0.1 + fm_loss_A * 0.9

            loss_D_A.backward()

            optimizer_D_A.step()

            # Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real, _ = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake, _ = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            loss_D_B = (loss_D_real + loss_D_fake)*0.5

            if args.fm_loss:
                pred_real, feats_real = netD_B(real_B)
                pred_fake, feats_fake = netD_B(fake_B.detach())

                fm_loss_B = get_fm_loss(feats_real, feats_fake, feat_criterion, args.cuda)

                loss_D_B = loss_D_B * 0.1 + fm_loss_B * 0.9

            loss_D_B.backward()

            optimizer_D_B.step()

            if iter % args.log_interval == 0:

                print('---------------------')
                print('GAN loss:', as_np(loss_GAN_A2B), as_np(loss_GAN_B2A))
                print('Identity loss:', as_np(loss_identity_A), as_np(loss_identity_B))
                print('Cycle loss:', as_np(loss_cycle_ABA), as_np(loss_cycle_BAB))
                print('D loss:', as_np(loss_D_A), as_np(loss_D_B))
                if args.fm_loss:
                    print('fm loss:', as_np(fm_loss_A), as_np(fm_loss_B))
                print('recon loss rate:', recon_loss_rate)
                print('time:', time.time() - prev_time)
                prev_time = time.time()

            if iter % args.plot_interval == 0:
                pass

            if iter % args.score_interval == 0:
                pass
                # print('Calculating score')
                #
                # for j, batch_ in enumerate(dataloader_test):
                #     real_A_test = Variable(input_A.copy_(batch_['A']))
                #     real_B_test = Variable(input_B.copy_(batch_['B']))
                #
                #     fake_AB_test = netG_A2B(real_A_test)
                #     fake_BA_test = netG_B2A(real_B_test)
                #     gan_metrics.save(real_A_test.detach(),
                #                      real_B_test.detach(),
                #                      fake_BA_test.detach(),
                #                      fake_AB_test.detach())
                #
                # score = gan_metrics.calculate_scores()
                # with open(csv_fn, 'a') as f:
                #     f.write('\n' + ','.join(str(e) for e in score))

            if iter % args.image_save_interval == 0:
                samples_path_ = os.path.join(samples_path, str(iter / args.image_save_interval))
                safe_mkdirs(samples_path_)

                # New savedir
                test_pth_AB = os.path.join(test_path, str(iter / args.image_save_interval), 'AB')
                test_pth_BA = os.path.join(test_path, str(iter / args.image_save_interval), 'BA')

                safe_mkdirs(test_pth_AB)
                safe_mkdirs(test_pth_BA)

                for j, batch_ in enumerate(dataloader_test):

                    real_A_test = Variable(input_A.copy_(batch_['A']))
                    real_B_test = Variable(input_B.copy_(batch_['B']))

                    fake_AB_test = netG_A2B(real_A_test)
                    fake_BA_test = netG_B2A(real_B_test)

                    if j < n_sample:
                        recovered_ABA_test = netG_B2A(fake_AB_test)
                        recovered_BAB_test = netG_A2B(fake_AB_test)

                        fn = os.path.join(samples_path_, str(j))
                        imageio.imwrite(fn + '.A.jpg', tensor2image(real_A_test[0], args.img_norm))
                        imageio.imwrite(fn + '.B.jpg', tensor2image(real_B_test[0], args.img_norm))
                        imageio.imwrite(fn + '.BA.jpg', tensor2image(fake_BA_test[0], args.img_norm))
                        imageio.imwrite(fn + '.AB.jpg', tensor2image(fake_AB_test[0], args.img_norm))
                        imageio.imwrite(fn + '.ABA.jpg', tensor2image(recovered_ABA_test[0], args.img_norm))
                        imageio.imwrite(fn + '.BAB.jpg', tensor2image(recovered_BAB_test[0], args.img_norm))

                    if j < n_test:
                        fn_A = os.path.basename(batch_['img_A'][0])
                        imageio.imwrite(os.path.join(test_pth_AB, fn_A), tensor2image(fake_AB_test[0], args.img_norm))

                        fn_B = os.path.basename(batch_['img_B'][0])
                        imageio.imwrite(os.path.join(test_pth_BA, fn_B), tensor2image(fake_BA_test[0], args.img_norm))

            if iter % args.model_save_interval == 0:
                # Save models checkpoints
                torch.save(netG_A2B.state_dict(), os.path.join(model_path, 'G_A2B_{}.pth'.format(iter)))
                torch.save(netG_B2A.state_dict(), os.path.join(model_path, 'G_B2A_{}.pth'.format(iter)))
                torch.save(netD_A.state_dict(), os.path.join(model_path, 'D_A_{}.pth'.format(iter)))
                torch.save(netD_B.state_dict(), os.path.join(model_path, 'D_B_{}.pth'.format(iter)))

            iter += 1

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')

    main(args)
