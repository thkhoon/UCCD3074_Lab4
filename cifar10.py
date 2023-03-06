import numpy as np
import torchvision
from torch.utils.data import Dataset

class CIFAR10(Dataset):
    
    def __init__(self, root = ".", train = True, download=False, transform=None, num_samples = None):
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.num_classes = 10          
        self.__transform = transform        
        self.__images  = []
        self.__targets = []

        dataset = torchvision.datasets.CIFAR10(root=root, 
                                      train=train, 
                                      download=download)

        if num_samples is not None:
            self.__num_samples = min(num_samples, len(dataset))
        else:
            self.__num_samples = len(self.__dataset)
        
        index = np.random.permutation(len(dataset))
        for i in range(self.__num_samples):
            self.__images.append(dataset.data[index[i]])
            self.__targets.append(dataset.targets[index[i]])
			
        self.__images   = np.array(self.__images)
        self.__targets  = np.array(self.__targets, dtype=np.int64)     
                       
    def __len__(self):
        return self.__num_samples
    
    def __getitem__(self, idx):
        
        img   = self.__images[idx]
        label = self.__targets[idx]

        if self.__transform:
            img = self.__transform(img)
            
        return img, label