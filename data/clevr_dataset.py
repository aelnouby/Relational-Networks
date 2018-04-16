"""Pytorch Dataset class implementation for CLEVR dataset
Author: Alaaeldin El-Nouby"""

import h5py
import cv2
import numpy as np
from torch.utils.data import Dataset
import json


class CLEVRDataset(Dataset):
    def __init__(self, file, transforms=None):
        self.file_path = file
        self.dataset = None

        file = h5py.File(self.file_path, 'r')
        self.len = int(np.array(file['len']))
        file.close()

        self.transforms = transforms
        self.h5py2int = lambda x: int(np.array(x))
        self.h5py2str = lambda x: str(np.array(x))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')

        example = self.dataset[str(idx)]

        image = cv2.imdecode(np.fromstring(example['image'][0], np.uint8), 1)
        questions = json.loads(self.h5py2str(example['questions']))
        answers = json.loads(self.h5py2str(example['answers']))

        qa_index = np.random.randint(0, len(questions))

        image = image.transpose(2, 0, 1)
        if self.transforms:
            image = self.transforms(image)

        image = (image / 255) - 0.5

        sample = {
            'image': image,
            'questions': questions[qa_index],
            'answers': answers[qa_index]
        }

        return sample
