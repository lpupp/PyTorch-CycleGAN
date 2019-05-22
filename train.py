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
parser.add_argument('--n_test', type=int, default=100, help='number of test samples to spit out')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='data/fashion/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

parser.add_argument('--upsample', action='store_true', help='If True: upsample; else: transposed 2D conv')
parser.add_argument('--keep_prop', action='store_true', help='Keep weights proportional to 3:64 ratio')
parser.add_argument('--G_extra', action='store_true', help='use extra layers in G')
parser.add_argument('--D_extra', action='store_true', help='use extra layers in D')
parser.add_argument('--slow_D', action='store_true', help='Slow the training of D to avoid mode collapse')
parser.add_argument('--recon_loss_acay', action='store_true', help='increase relative importance of recon loss')
parser.add_argument('--recon_acay_rate', type=float, default=2, help='increase relative importance of recon loss (rate)')

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


def tensor2image(tensor):
    image = 127.5*(as_np(tensor) + 1.0)
    return image.astype(np.uint8).transpose(1, 2, 0)


def as_np(tensor):
    return tensor.cpu().float().detach().numpy()


def get_fm_loss(real_feats, fake_feats, criterion, cuda):
    # TODO(lpupp) need to output discriminator intermediates and find best way
    # to weight fm_loss in total loss.
    losses = 0
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        if cuda:
            loss = criterion(l2, Variable(torch.ones(l2.size())).cuda())
        else:
            loss = criterion(l2, Variable(torch.ones(l2.size())))
        losses += loss

    return losses


def main(args):
    torch.manual_seed(0)

    modelarch = 'C_{0}_{1}_{2}{3}{4}{5}{6}{7}{8}'.format(args.size, args.batch_size, args.lr,  #0, 1, 2
                                                         '_' if args.G_extra or args.D_extra else '',  # 3
                                                         'G' if args.G_extra else '',  # 4
                                                         'D' if args.D_extra else '',  # 5
                                                         '_U' if args.upsample else '',  # 6
                                                         '_S' if args.slow_D else '',  # 7
                                                         '_RL' if args.recon_loss_acay else '')  # 8

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
    netD_A = Discriminator(args.input_nc, extra_layer=args.D_extra)
    netD_B = Discriminator(args.output_nc, extra_layer=args.D_extra)

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
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    #feat_criterion = nn.HingeEmbeddingLoss()  # TODO

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

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    transforms_ = []

    if args.resize_crop:
        transforms_ += [transforms.Resize(int(args.size*1.12), Image.BICUBIC),
                        transforms.RandomCrop(args.size)]
    else:
        transforms_ += [transforms.Resize(args.size, Image.BICUBIC)]

    if args.horizontal_flip:
        transforms_ += [transforms.RandomHorizontalFlip()]

    transforms_ += [transforms.ToTensor()]

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

    transforms_test_ = [transforms.Resize(args.size, Image.BICUBIC),
                        transforms.ToTensor()]
    transforms_test_ += transforms_norm

    dataloader_test = DataLoader(ImageDataset(args.dataroot, transforms_=transforms_test_, mode='test'),
                                 batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)
    # Training ######
    iter = 0
    prev_time = time.time()

    #gan_metrics = ConvNetFeatureSaver()
    #csv_fn = os.path.join(args.output_dir, modelarch, modelarch + '.csv')

    recon_loss_acay_trigger = int(args.decay_epoch * 0.5)
    for epoch in range(args.load_iter, args.n_epochs):

        recon_loss_rate = 1.0
        if args.recon_loss_acay:
            if epoch > recon_loss_acay_trigger:
                effective_epoch = (epoch - recon_loss_acay_trigger)
                # the recon_loss_rate maxes out before n_epoch (because of "- args.decay_epoch")
                recon_loss_rate = 1.0 + (effective_epoch/(args.n_epochs - args.decay_epoch)) * (args.recon_acay_rate - 1)

        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            # Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * 5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + (loss_cycle_ABA + loss_cycle_BAB) * recon_loss_rate
            loss_G.backward()

            optimizer_G.step()

            # Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            #pred_real, feats_real = netD_A(real_A)
            #pred_fake, feats_fake = netD_A(fake_A.detach())

            #fm_loss = get_fm_loss(feats_real, feats_fake, feat_criterion, args.cuda)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            #loss_D_A = (loss_D_real + loss_D_fake)*0.5 + fm_loss*0.9  # TODO
            loss_D_A.backward()

            optimizer_D_A.step()

            # Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            #pred_real, feats_real = netD_B(real_B)
            #pred_fake, feats_fake = netD_B(fake_B.detach())

            #fm_loss = get_fm_loss(feats_real, feats_fake, feat_criterion, args.cuda)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            #loss_D_B = (loss_D_real + loss_D_fake)*0.5 + fm_loss*0.9  # TODO
            loss_D_B.backward()

            optimizer_D_B.step()

            if iter % args.log_interval == 0:

                print("---------------------")
                print("GAN loss:", as_np(loss_GAN_A2B), as_np(loss_GAN_B2A))
                print("Identity loss:", as_np(loss_identity_A), as_np(loss_identity_B))
                print("Cycle loss:", as_np(loss_cycle_ABA), as_np(loss_cycle_BAB))
                print("D loss:", as_np(loss_D_A), as_np(loss_D_B))
                print("recon loss rate:", recon_loss_rate)
                print("time:", time.time() - prev_time)
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

                        break

                    real_A_test = Variable(input_A.copy_(batch_['A']))
                    real_B_test = Variable(input_B.copy_(batch_['B']))

                    fake_AB_test = netG_A2B(real_A_test)
                    fake_BA_test = netG_B2A(real_B_test)

                    if j < args.n_test:
                        recovered_ABA_test = netG_B2A(fake_AB_test)
                        recovered_BAB_test = netG_A2B(fake_AB_test)

                        fn = os.path.join(samples_path_, str(j))
                        imageio.imwrite(fn + '.A.jpg', tensor2image(real_A_test[0]))
                        imageio.imwrite(fn + '.B.jpg', tensor2image(real_B_test[0]))
                        imageio.imwrite(fn + '.BA.jpg', tensor2image(fake_BA_test[0]))
                        imageio.imwrite(fn + '.AB.jpg', tensor2image(fake_AB_test[0]))
                        imageio.imwrite(fn + '.ABA.jpg', tensor2image(recovered_ABA_test[0]))
                        imageio.imwrite(fn + '.BAB.jpg', tensor2image(recovered_BAB_test[0]))

                    fn_A = os.path.basename(batch_['img_A'][0])
                    imageio.imwrite(os.path.join(test_pth_AB, fn_A), tensor2image(fake_AB_test[0]))

                    fn_B = os.path.basename(batch_['img_B'][0])
                    imageio.imwrite(os.path.join(test_pth_BA, fn_B), tensor2image(fake_BA_test[0]))

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


if __name__ == "__main__":
    global args
    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')

    main(args)
