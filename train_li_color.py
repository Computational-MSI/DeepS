from __future__ import print_function
import argparse

import torch.nn as nn

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.autograd import Variable
import dataload_color as dataload
import os
from model_unet_color import _netlocalD, _netG, opt
import cv2
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from mask_gen_color import gen_mask
import matplotlib.pyplot as plt

resume_epoch = 0
opt.cuda = True
try:
    os.makedirs("4_6/rec/0.8/train/cropped")
    os.makedirs("4_6/rec/0.8/train/real")
    os.makedirs("4_6/rec/0.8/train/recon")
    os.makedirs("4_6/rec/0.8")
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)


cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


# print(opt)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# paragram
opt.netG='3_25/rec/0.8/[1,2,4,8,16]/netG_streetview.pth'
opt.netD='3_25/rec/0.8/[1,2,4,8,16]/netlocalD.pth'
opt.batchSize=8
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3
nef = int(opt.nef)
opt.lr=0.0002

nBottleneck = int(opt.nBottleneck)
wtl2 = float(opt.wtl2)
overlapL2Weight = 10
k_value = [1,2, 4, 8, 16]
###networks
netG = _netG(opt)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']

netD = _netlocalD(opt)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD)['epoch']

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD,step_size=5,gamma = 0.8)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG,step_size=5,gamma = 0.8)

# criterion
criterion = nn.BCELoss()
criterionL1 = nn.L1Loss()

# data
train_Dataset = np.array(dataload.TextileData(root='/root/autodl-tmp/VAE/picture'))

train_loader = DataLoader(dataset=train_Dataset, batch_size=opt.batchSize, shuffle=True, drop_last=True,num_workers=12,pin_memory=True)

epoches = 10000
label = torch.FloatTensor(opt.batchSize,1)
real_label = 1
fake_label = 0

input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
real_center = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

one = torch.FloatTensor([1])

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterionL1.cuda()
    one = one.cuda()
    input_real, input_cropped, label = input_real.cuda(), input_cropped.cuda(), label.cuda()
    real_center = real_center.cuda()

input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
label = Variable(label)

print(label.shape)
real_center = Variable(real_center)
Ms_generator = gen_mask(k_value,opt.batchSize,opt.imageSize, 0.8)




loss_init = 500000
opt.niter=resume_epoch+10
for epoch in range(resume_epoch, opt.niter):
    print('start epoch')
    epoch_loss = 0

    for i, data in enumerate(train_loader, 0):

        real_cpu = data

        if data.shape[0] != opt.batchSize:
            break
        # print(data.shape)

        batch_size = real_cpu.size(0)

        input_real.data.copy_(real_cpu)
        #print(np.shape(next(Ms_generator)))

        Ms = np.array(next(Ms_generator)).reshape([opt.batchSize, 3, opt.imageSize, opt.imageSize])
        Ms_fan=np.where(Ms>0,0,1)
        Ms_=Ms_fan.copy()
        A=real_cpu * Ms_fan
        Ms_[:,0,:,:]=np.where(A[:,0,:,:]>0,0,0)
        Ms_[:,1,:,:]=np.where(A[:,1,:,:]>0,1,0)
        Ms_[:,2,:,:]=np.where(A[:,2,:,:]>0,0,0)

        input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu * Ms+Ms_)

        if opt.cuda:
            real_gpu = data.cuda()
            input_cropped = input_cropped.cuda()
            real_center = real_center.cuda()

        # train with real
        netD.zero_grad()
        label.data.fill_(real_label)
        output = netD(input_real)
        # print(output.shape)
        # print(label.shape)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        # noise.data.resize_(batch_size, nz, 1, 1)
        # noise.data.normal_(0, 1)
        fake = netG(input_cropped)
        label.data.fill_(fake_label)
        # print(input_cropped.shape)
        # print(fake.shape)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG_D = criterion(output, label)

        
        errG_l1 = criterionL1(fake, input_real)
        errG_l1 = errG_l1.mean()

        errG = (1 - wtl2) * errG_D + wtl2 * errG_l1

        errG.backward()

        D_G_z2 = output.data.mean()
        optimizerG.step()
        if i%10==0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
                   % (epoch, opt.niter, i, len(train_loader),
                     errD.item(), errG_D.item(), errG_l1.item(), D_x, D_G_z1,))
        #if i==len(train_loader)-1:
        
        epoch_loss = epoch_loss+errG
        if D_G_z1/D_x<0.01:
            print(D_G_z1/D_x)
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)
           
        if  i!=0 and i % (len(train_loader)-1)==0:
            vutils.save_image(real_cpu,
                              '4_6/rec/0.8/train/real/real_samples_epoch_%03d.png' % (epoch))
            vutils.save_image(input_cropped.data,
                              '4_6/rec/0.8/train/cropped/cropped_samples_epoch_%03d.png' % (epoch))
            recon_image = fake.data
            vutils.save_image(recon_image,
                              '4_6/rec/0.8/train/recon/recon_center_samples_epoch_%03d.png' % (epoch))
    schedulerD.step()
    schedulerG.step()
    if epoch_loss < loss_init:
        loss_init = epoch_loss
        print(epoch)
        # do checkpointing
        torch.save({'epoch': epoch + 1,
                   'state_dict': netG.state_dict()},
                   '4_6/rec/0.8/netG_streetview.pth')
        torch.save({'epoch': epoch + 1,
                    'state_dict': netD.state_dict()},
                    '4_6/rec/0.8/netlocalD.pth')