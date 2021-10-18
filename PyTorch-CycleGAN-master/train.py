from utils import PARSER
import itertools
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import matplotlib.pyplot as plt
from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from tqdm import tqdm
#from utils import Logger
import logging
from utils import weights_init_normal
from datasets import ImageDataset
import time
import warnings
warnings.filterwarnings('ignore')
if __name__=='__main__':
    parser = PARSER()
    parser.add_argument('--batchSize', type=int, default=2, help='size of the batches')
    parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=128, help='number of epochs of training')
    parser.add_argument('--lr', type=float, default=0.0008, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=8,
                      help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--save_period', type=int, default=8, help='save model per epoch period')
    opt = parser.parse_args()
    print('-'*32)
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)
    netD_A = Discriminator(opt.input_nc)
    netD_B = Discriminator(opt.output_nc)

    if opt.cuda:
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

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                    lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    print('net initalate done...')

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batchSize,1).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize,1).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
                    transforms.RandomCrop(opt.size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    transforms_=[transforms.ToTensor()]#transforms.Resize(opt.size, Image.BICUBIC),,transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=False),
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu,drop_last=True)
    print('data loading done...')

    # Loss plot
    #logger = Logger(opt.n_epochs, len(dataloader))
    logger=logging.getLogger('horse2zerbo')
    ###################################
    loss={'G':[],'D_A':[],'D_B':[]}
    loss_mean = {'G': [], 'D_A': [], 'D_B': []}

    #time.sleep(3)
    ###### Training ######
    print('training...')
    for epoch in range(opt.epoch, opt.n_epochs+1):
        t1=time.time()
        for i, batch in enumerate(dataloader):
            # Set model input
            #print(epoch,'--',i,'**')
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*5.0

            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*5.0
            #loss['identity_B']=loss_identity_B
            #loss['identity_A'] = loss_identity_A

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            loss['G'].append(loss_G.item())
            optimizer_G.step()

            ###################################
            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()
            loss['D_A'].append(loss_D_A.item())
            optimizer_D_A.step()

            ###################################
            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()
            loss['D_B'].append(loss_D_B.item())
            optimizer_D_B.step()

            ###################################
            # Progress report (http://localhost:8097)
            #logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        t2=time.time()
        for key_loss in loss_mean:
            loss_mean[key_loss].append(np.mean(loss[key_loss][-i:]))
        info = ''
        for loss_name, loss_value in loss_mean.items():
            info += loss_name + ':%.3f\t' % loss_value[-1]
        info+='eplased:%d s'%int(t2-t1)
        logger.info(info)
        print('\a epoch%3d---'%epoch, info)

        if epoch%opt.save_period==0:
            # Save models checkpoints
            torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
            torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
            torch.save(netD_A.state_dict(), 'output/netD_A.pth')
            torch.save(netD_B.state_dict(), 'output/netD_B.pth')
            print('model of epoch %d saved'%epoch)
    ###################################

    for loss_ in loss_mean:
        np.save('output/Mean-%s.npy'%loss_,np.array(loss_mean[loss_]))
        np.save('output/%s.npy' % loss_, np.array(loss[loss_]))
        plt.plot(loss_mean[loss_],label=loss_)
    plt.legend()
    plt.show()

    for loss_ in loss_mean:
        plt.plot(loss_mean[loss_])
        plt.show()
