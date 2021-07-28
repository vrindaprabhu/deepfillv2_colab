import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import *


class InpaintDataset(Dataset):
    def __init__(self):
        self.imglist = [INIMAGE]
        self.masklist = [MASKIMAGE]
        self.setsize = RESIZE_TO

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        # image
        img = cv2.imread(self.imglist[index])
        mask = cv2.imread(self.masklist[index])[:, :, 0]
        ## COMMENTING FOR NOW
        # h, w = mask.shape
        # # img = cv2.resize(img, (w, h))
        img = cv2.resize(img, self.setsize)
        mask = cv2.resize(mask, self.setsize)
        ##
        # find the Minimum bounding rectangle in the mask
        """
        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cidx, cnt in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(cnt)
            mask[y:y+h, x:x+w] = 255
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = (
            torch.from_numpy(img.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .contiguous()
        )
        mask = (
            torch.from_numpy(mask.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .contiguous()
        )
        return img, mask
