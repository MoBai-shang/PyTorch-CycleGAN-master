import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=True, mode='train',image_format='jpg'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.'+image_format))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.'+image_format))
        print('%s data size:'%mode,self.__len__())
    def __getitem__(self, index):
        name_A=self.files_A[index % len(self.files_A)]
        if self.unaligned:
            name_B = self.files_B[random.randint(0, len(self.files_B) - 1)]
        else:
            name_B = self.files_B[index % len(self.files_B)]
        item_A = self.transform(Image.open(name_A))
        item_B = self.transform(Image.open(name_B))
        #if self.unaligned:
        for i in range(1000):
            if item_B.shape[0]!=3:
                name_B=self.files_B[random.randint(0, len(self.files_B) - 1)]
                item_B = self.transform(Image.open(name_B))
            else:
                break

        #if not item_A.shape==item_B.shape==(3,self.image_size,self.image_size):
        #    print(item_A.shape,item_B.shape,self.files_A[index % len(self.files_A)],self.files_B[index % len(self.files_B)])
        return {'A': item_A, 'B': item_B,'A_name':os.path.basename(name_A).split('.')[0],'B_name':os.path.basename(name_B).split('.')[0]}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))