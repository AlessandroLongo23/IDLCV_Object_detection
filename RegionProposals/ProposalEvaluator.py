import os
import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.patches as patches
from tqdm import tqdm
import cv2

from RegionProposals.ProposalExtractor import ProposalExtractor
from Potholes.DatasetAnalyzer import DatasetAnalyzer

from utils.globalConst import *

class ProposalEvaluator:
    def __init__(self):
        self.dataset_analyzer = DatasetAnalyzer()
        self.proposal_extractor = ProposalExtractor()
        pass
    
    def compute_iou(self, boxA, boxB):
        xA = max(boxA['xmin'], boxB['xmin'])
        yA = max(boxA['ymin'], boxB['ymin'])
        xB = min(boxA['xmax'], boxB['xmax'])
        yB = min(boxA['ymax'], boxB['ymax'])

        interWidth = max(0, xB - xA + 1)
        interHeight = max(0, yB - yA + 1)
        intersection = interWidth * interHeight

        boxAArea = (boxA['xmax'] - boxA['xmin'] + 1) * (boxA['ymax'] - boxA['ymin'] + 1)
        boxBArea = (boxB['xmax'] - boxB['xmin'] + 1) * (boxB['ymax'] - boxB['ymin'] + 1)

        union = boxAArea + boxBArea - intersection
        return intersection / union if union > 0 else 0
    
    def evaluate_proposals(self, alg, num_proposals_list, iou_threshold=0.5):
        num_proposals_list = sorted(num_proposals_list)
        results = {'recalls': [], 'mabo': []}

        for n in num_proposals_list:
            recall_numerator = 0
            mabo_total = 0.0
            total_ground_truth_boxes = 0

            for split in ['train', 'val', 'test']:
                for idx in [folder.split('-')[1] for folder in os.listdir(f'{ROOT_DIR}/{split}')]:
                    ground_truth_boxes = self.dataset_analyzer.parse_annotation(split, idx)
                    total_ground_truth_boxes += len(ground_truth_boxes)
                    proposals = json.load(open(f'{ROOT_DIR}/{split}/img-{idx}/{alg}.json', 'r'))
                    proposals = proposals[:n]

                    matched_ground_truth_boxes = 0
                    for ground_truth_box in ground_truth_boxes:
                        if len(proposals) > 0:
                            max_iou = max([self.compute_iou(ground_truth_box, proposal) for proposal in proposals])
                        else:
                            max_iou = 0.0
                            
                        mabo_total += max_iou
                        if max_iou >= iou_threshold:
                            matched_ground_truth_boxes += 1

                    recall_numerator += matched_ground_truth_boxes

            # Calculate recall and MABO for n proposals
            recall = recall_numerator / total_ground_truth_boxes if total_ground_truth_boxes > 0 else 0
            mabo = mabo_total / total_ground_truth_boxes if total_ground_truth_boxes > 0 else 0

            results['recalls'].append(recall)
            results['mabo'].append(mabo)

            print(f"{alg} - Proposals: {n}, Recall: {recall:.4f}, MABO: {mabo:.4f}")

        return results
    
    def plot_metric_vs_proposals(self, num_proposals_list, metric_ss, metric_eb, metric_name):
        plt.figure(figsize=(10, 6))
        plt.plot(num_proposals_list, metric_ss, marker='o', label='Selective Search')
        plt.plot(num_proposals_list, metric_eb, marker='s', label='Edge Boxes')
        plt.xlabel('Number of Proposals')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} vs. Number of Proposals')
        plt.legend()
        plt.grid(True)
        plt.xticks(num_proposals_list)
        plt.ylim(0, 1.05)
        plt.show()
        
    def display_ground_truth_and_proposals(self, idx, alg, top_n=100, save=False):
        image = cv2.imread(f'{TRAIN_DIR}/img-{idx}/original.jpg')
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(image)
        
        ground_truth_boxes = self.dataset_analyzer.parse_annotation('train', idx)
        for ground_truth_box in ground_truth_boxes:
            rect = patches.Rectangle(
                (ground_truth_box['xmin'], ground_truth_box['ymin']),
                ground_truth_box['xmax'] - ground_truth_box['xmin'],
                ground_truth_box['ymax'] - ground_truth_box['ymin'],
                linewidth=2, edgecolor='blue', facecolor='none', label='Ground Truth'
            )
            ax.add_patch(rect)

        proposals = json.load(open(f'{TRAIN_DIR}/img-{idx}/{alg}.json'))
        for prop in proposals[:top_n]:
            rect = patches.Rectangle(
                (prop['xmin'], prop['ymin']),
                prop['xmax'] - prop['xmin'],
                prop['ymax'] - prop['ymin'],
                linewidth=1, edgecolor='red', facecolor='none', label='Proposal'
            )
            ax.add_patch(rect)

        # Avoid duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        plt.title(f'Ground Truth vs. {alg} Proposals')
        plt.axis('off')
        plt.show()

        if save:
            fig.savefig(f'{TRAIN_DIR}/ground_truth_vs_{alg}_proposals', bbox_inches='tight')
            
    def assign_labels_to_proposals(self, k_1=0.3, k_2=0.7):
        for alg in ['selective_search', 'edge_boxes']:
            for split in ['train', 'val', 'test']:
                for img_folder in tqdm(os.listdir(f'{ROOT_DIR}/{split}'), desc=f'Assigning labels to {alg} proposals'):
                    idx = int(img_folder.split('-')[1])
                    ground_truth_boxes = self.dataset_analyzer.parse_annotation(split, idx)
                    proposals = json.load(open(f'{ROOT_DIR}/{split}/img-{idx}/{alg}.json', 'r'))

                    for proposal in proposals:
                        max_iou = max([self.compute_iou(ground_truth_box, proposal) for ground_truth_box in ground_truth_boxes])

                        if max_iou >= k_2:
                            proposal['label'] = 'pothole'
                        elif max_iou < k_1:
                            proposal['label'] = 'background'
                        else:
                            proposal['label'] = 'NaN'
                            
                    with open(f'{ROOT_DIR}/{split}/img-{idx}/{alg}.json', 'w') as f:
                        json.dump(proposals, f, indent=4)

    def visualize_labeled_proposals(self, idx, alg, save=False):
        image = cv2.imread(f'{TRAIN_DIR}/img-{idx}/original.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(image)
        
        proposals = json.load(open(f'{TRAIN_DIR}/img-{idx}/{alg}.json', 'r'))

        for proposal in [p for p in proposals if p['label'] == 'pothole']:
            xmin, ymin, xmax, ymax = proposal['xmin'], proposal['ymin'], proposal['xmax'], proposal['ymax']
            width = xmax - xmin
            height = ymax - ymin

            ax.add_patch(patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='green', facecolor='none'))
            ax.text(xmin, ymin, proposal['label'], fontsize=8, color='green', bbox=dict(facecolor='green', alpha=0.5))

        plt.title('Labeled Proposals')
        plt.axis('off')
        plt.show()

        if save:
            fig.savefig(f'{TRAIN_DIR}/img-{idx}/labeled_proposals.jpg', bbox_inches='tight')
            
    def apply_nms(self, idx, alg, nms_threshold=0.3):
        proposals = json.load(open(f'{TRAIN_DIR}/img-{idx}/{alg}.json', 'r'))
        
        boxes = [[
            proposal['xmin'], 
            proposal['ymin'], 
            proposal['xmax'] - proposal['xmin'], 
            proposal['ymax'] - proposal['ymin']
        ] for proposal in proposals]

        scores = [1.0] * len(boxes)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=nms_threshold)

        filtered_proposals = []
        if len(indices) > 0:
            for idx in indices:
                if isinstance(idx, (list, tuple, np.ndarray)):
                    index = idx[0]
                else:
                    index = idx
                    
                if index < len(proposals):
                    filtered_proposals.append(proposals[index])
                else:
                    print(f"Index {index} out of bounds for proposals list with length {len(proposals)}.")

        return filtered_proposals