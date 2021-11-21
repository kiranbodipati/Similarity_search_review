import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class SiameseNetworkDataset(Dataset):
    def __init__(self, pairs_dir, transform=None):
        self.pairs_dir = pairs_dir
        self.transform = transform
        self.pairs_df = pd.read_csv(pairs_dir)

    def __len__(self):
        return len(self.pairs_df)
    
    def __getitem__(self, index):
        row = self.pairs_df.iloc[index]
        img0 = Image.open(row['inputA']).convert("RGB")
        img1 = Image.open(row['inputB']).convert("RGB")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        # return img0, img1 , torch.from_numpy(np.array([row['label']],dtype=np.float32))
        return img0, img1, row['label']
