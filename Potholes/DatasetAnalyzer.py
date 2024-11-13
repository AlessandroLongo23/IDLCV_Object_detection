import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2

from utils.globalConst import *

class DatasetAnalyzer:
    def __init__(self):
        pass
    
    def _map(self, v, start_min, start_max, end_min, end_max):
        return int((v - start_min) * (end_max - end_min) / (start_max - start_min) + end_min)
    
    def parse_annotation(self, split, idx):
        tree = ET.parse(f'{ROOT_DIR}/{split}/img-{idx}/ground_truth.xml')
        root = tree.getroot()
        
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        
        boxes = []
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            box = {
                'label': obj.find('name').text,
                'xmin': self._map(int(bbox.find('xmin').text), 0, width, 0, IMG_SIZE),
                'ymin': self._map(int(bbox.find('ymin').text), 0, height, 0, IMG_SIZE),
                'xmax': self._map(int(bbox.find('xmax').text), 0, width, 0, IMG_SIZE),
                'ymax': self._map(int(bbox.find('ymax').text), 0, height, 0, IMG_SIZE),
            }
            boxes.append(box)

        return boxes
    
    def visualize_image(self, idx, boxes, ax):
        image = cv2.imread(f'{TRAIN_DIR}/img-{idx}/original.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        ax.imshow(image)

        for box in boxes:
            xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, box['label'], fontsize=12, color='red', weight='bold')

        ax.axis('off')
        
    def visualize_image_with_annotations(self, split, idx):
        image = cv2.imread(f'{ROOT_DIR}/{split}/img-{idx}/original.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        boxes = self.parse_annotation(split, idx)

        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(image)
        ax.set_title(f"Image: img-{idx} | Potholes: {len(boxes)}")
        ax.axis('off')

        for box in boxes:
            xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        plt.show()
        
    def visualize_samples(self, split, num_samples=6, rows=2, cols=3):
        img_folders = os.listdir(f'{ROOT_DIR}/{split}')  

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        fig.text(0.5, 0.02, "Annotated Pothole Images", ha='center', fontsize=16)

        axes = axes.flatten()

        for i in range(num_samples):
            idx = img_folders[i].split('-')[1]
            boxes = self.parse_annotation(split, idx)
            self.visualize_image(idx, boxes, axes[i])
            axes[i].set_title(f"img-{idx}", fontsize=10)

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()
        
    def get_unique_image_sizes_and_ratios(self):
        size_counts = {}
        aspect_ratios = set()
        for image_file in os.listdir(TRAIN_DIR):
            image_path = f'{TRAIN_DIR}/{image_file}/original.jpg'
            with Image.open(image_path) as img:
                width, height = img.size
                size = (width, height)

                if height != 0:
                    ratio = width / height
                    aspect_ratios.add(round(ratio, 4))  # Rounded for readability
                else:
                    print(f"Warning: Image {image_file} has zero height.")
                    ratio = None

                if size in size_counts:
                    size_counts[size] += 1
                else:
                    size_counts[size] = 1
                    
        return size_counts, aspect_ratios
    
    def count_annotations(self):
        total_annotations = 0
        objects_per_image = {}

        for image_path in os.listdir(TRAIN_DIR):
            tree = ET.parse(f'{TRAIN_DIR}/{image_path}/ground_truth.xml')
            root = tree.getroot()
            num_objects = len(root.findall('object'))
            total_annotations += num_objects
            
            img_idx = image_path.split('-')[1]
            objects_per_image[img_idx] = num_objects

        return total_annotations, objects_per_image
    
    def find_images_with_max_objects(self, objects_per_image):
        max_objects = max(objects_per_image.values())
        image_indexes = [idx for idx, count in objects_per_image.items() if count == max_objects]

        return max_objects, image_indexes