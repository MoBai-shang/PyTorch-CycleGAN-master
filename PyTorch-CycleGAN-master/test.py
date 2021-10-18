#!/usr/bin/python3

from utils import PARSER
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset
from PIL import Image
if __name__=='__main__':
    parser = PARSER()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')

    parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)

    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
    netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    # Dataset loader
    transforms_ = [ transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    transforms_=[transforms.Resize(opt.size, Image.BICUBIC),transforms.ToTensor()]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'),
                            batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
    ###################################

    ###### Testing######

    # Create output dirs if they don't exist
    if not os.path.exists('output/A'):
        os.makedirs('output/A')
    if not os.path.exists('output/B'):
        os.makedirs('output/B')

    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # Generate output
        #fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
        #fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

        fake_B = netG_A2B(real_A).data
        fake_A = netG_B2A(real_B).data

        for G_A,G_B,src_A,src_B in zip(fake_A,fake_B,batch['A_name'],batch['B_name']):
            # Save image files
            save_image(G_A, 'output/A/G-%s.png' %src_B)
            save_image(G_B, 'output/B/G-%s.png' % src_A)

        sys.stdout.write('\rGenerated images %04d batch of %04d' % ((i+1), len(dataloader)))

    sys.stdout.write('\n')
    ###################################
