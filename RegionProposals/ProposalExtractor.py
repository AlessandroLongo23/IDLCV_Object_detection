from tqdm import tqdm 
import json
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.globalConst import *

class ProposalExtractor:
    def selective_search(self, image):
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        return rects
                
    def edge_boxes(self, image):
        edge_detector = cv2.ximgproc.createStructuredEdgeDetection('./utils/model.yml.gz')
        eb = cv2.ximgproc.createEdgeBoxes()
        eb.setMaxBoxes(2000)

        edges = edge_detector.detectEdges(np.float32(image) / 255.0)
        orientation = edge_detector.computeOrientation(edges)
        edges_nms = edge_detector.edgesNms(edges, orientation)

        rects, _ = eb.getBoundingBoxes(edges_nms, orientation)
        return rects

    def extract_with(self, alg):
        for split in ['train', 'val', 'test']:
            for image_file in tqdm(os.listdir(f'{ROOT_DIR}/{split}'), desc=f'Extracting {split} proposals with {alg}'):
                image = cv2.imread(f'{ROOT_DIR}/{split}/{image_file}/original.jpg')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                
                extract_alg = self.edge_boxes if alg == 'edge_boxes' else self.selective_search

                rects = [{
                    'label': '',
                    'xmin': int(rect[0]),
                    'ymin': int(rect[1]),
                    'xmax': int(rect[2]),
                    'ymax': int(rect[3]),
                } for rect in extract_alg(image)]
                
                with open(f'{ROOT_DIR}/{split}/{image_file}/{alg}.json', 'w') as f:
                    json.dump(rects, f, indent=4)
                
    def draw_bounding_boxes(self, image, boxes, color=(0, 255, 0), thickness=2):
        for box in boxes:
            xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

        return image
    
    def visualize_top_proposals(self, idx, alg, top_n, save=False):
        image = cv2.imread(f'{TRAIN_DIR}/img-{idx}/original.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        proposals = json.load(open(f'{TRAIN_DIR}/img-{idx}/{alg}.json', 'r'))
        top_proposals = proposals[:top_n]

        image_with_boxes = image.copy()
        image_with_boxes = self.draw_bounding_boxes(image_with_boxes, top_proposals, color=(255, 0, 0), thickness=1)

        # Display the image
        plt.figure(figsize=(6, 6))
        plt.imshow(image_with_boxes)
        plt.title(f"Top {top_n} {alg} Proposals")
        plt.axis('off')
        plt.show()

        if save:
            cv2.imwrite(f'{TRAIN_DIR}/img-{idx}/{alg}_top_{top_n}.jpg', image_with_boxes)    