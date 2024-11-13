import json
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import os

from utils.globalConst import *

class ProposalDataset(Dataset):
    def __init__(self, split, label_encoder, transform=None):
        self.split = split
        self.transform = transform
        self.label_encoder = label_encoder

        self.samples = []
        for img_folder in os.listdir(f'{ROOT_DIR}/{self.split}'): 
            image_path = f'{ROOT_DIR}/{self.split}/{img_folder}/original.jpg'
            
            for alg in ['selective_search', 'edge_boxes']:
                labeled_proposals = json.load(open(f'{ROOT_DIR}/{self.split}/{img_folder}/{alg}.json', 'r'))
                for proposal in labeled_proposals:
                    box = [proposal['xmin'], proposal['ymin'], proposal['xmax'], proposal['ymax']]
                    label = proposal['label']
                    self.samples.append({'image_path': image_path, 'box': box, 'label': label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with Image.open(sample['image_path']) as img:
            x, y, w, h = sample['box']
            cropped_img = img.crop((x, y, x + w, y + h)).convert('RGB')
            if self.transform:
                cropped_img = self.transform(cropped_img)
            else:
                cropped_img = cropped_img.resize((IMG_SIZE, IMG_SIZE))
                cropped_img = np.array(cropped_img).astype(np.float32) / 255.0
                cropped_img = np.transpose(cropped_img, (2, 0, 1))
                cropped_img = torch.tensor(cropped_img)

        label_encoded = self.label_encoder.transform([sample['label']])[0]
        label_encoded = torch.tensor(label_encoded, dtype=torch.long)

        return cropped_img, label_encoded