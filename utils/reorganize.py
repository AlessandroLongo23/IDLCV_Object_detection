import os
import json
import shutil
import numpy as np

from utils.globalConst import *

def reorganize_in_folders():
    source_folder = "Potholes/annotated-images"

    for filename in os.listdir(source_folder):
        if filename.endswith(".jpg"):
            base_name = os.path.splitext(filename)[0]
            xml_file = f"{base_name}.xml"
            
            if os.path.isfile(os.path.join(source_folder, xml_file)):
                subfolder = os.path.join(source_folder, base_name)
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder)
                
                jpg_file = os.path.join(source_folder, filename)
                xml_file = os.path.join(source_folder, xml_file)
                shutil.move(jpg_file, os.path.join(subfolder, "original.jpg"))
                shutil.move(xml_file, os.path.join(subfolder, "data.xml"))
                print(f"Moved {filename} and {xml_file} to {subfolder}")
            else:
                print(f"Skipping {filename} as the XML file {xml_file} was not found.")

def split_in_folders():
    splits = json.load(open("./Potholes/splits.json"))
    source_folder = './Potholes/annotated-images'

    train_folder = os.path.join(source_folder, 'train')
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    for folder in [split.split('.')[0] for split in splits['train']]:
        folder_path = os.path.join(source_folder, folder)
        shutil.move(folder_path, os.path.join(train_folder))
        
    test_folder = os.path.join(source_folder, 'test')
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    for folder in [split.split('.')[0] for split in splits['test']]:
        folder_path = os.path.join(source_folder, folder)
        shutil.move(folder_path, os.path.join(test_folder))
        
def delete_data_json():
    for folder in os.listdir('{TEST_DIR}'):
        if os.path.exists(f'{TEST_DIR}/{folder}/data.json'):
            os.remove(f'{TEST_DIR}/{folder}/data.json')
            
def delete_proposals_folder():
    for folder in os.listdir('{TRAIN_DIR}'):
        if os.path.exists(f'{TRAIN_DIR}/{folder}/proposals'):
            shutil.rmtree(f'{TRAIN_DIR}/{folder}/proposals')
            
def rename_data_to_ground_truth():
    for split in ['train', 'test']:
        for folder in os.listdir(f'./Potholes/annotated-images/{split}'):
            os.rename(f'./Potholes/annotated-images/{split}/{folder}/data.xml', f'./Potholes/annotated-images/{split}/{folder}/ground_truth.xml')

def reorganize_splits():
    splits = json.load(open('./Potholes/splits_origin.json'))
    
    train_ids = splits.get('train', [])
    test_ids = splits.get('test', [])
    
    np.random.shuffle(train_ids)
    val_ids = train_ids[:int(len(train_ids) * 0.25)]
    train_ids = train_ids[int(len(train_ids) * 0.25):]
    
    train_ids = [int(idx.split('.')[0].split('-')[-1]) for idx in train_ids]
    val_ids = [int(idx.split('.')[0].split('-')[-1]) for idx in val_ids]
    test_ids = [int(idx.split('.')[0].split('-')[-1]) for idx in test_ids]
    
    train_ids.sort()
    val_ids.sort()
    test_ids.sort()
    
    for folder in os.listdir(TRAIN_DIR):
        if int(folder.split('-')[-1]) in val_ids:
            os.rename(f'{TRAIN_DIR}/{folder}', f'{VAL_DIR}/{folder}')
        elif int(folder.split('-')[-1]) in test_ids:
            os.rename(f'{TRAIN_DIR}/{folder}', f'{TEST_DIR}/{folder}')
        
    for folder in os.listdir(VAL_DIR):
        if int(folder.split('-')[-1]) in train_ids:
            os.rename(f'{VAL_DIR}/{folder}', f'{TRAIN_DIR}/{folder}')
        elif int(folder.split('-')[-1]) in test_ids:
            os.rename(f'{VAL_DIR}/{folder}', f'{TEST_DIR}/{folder}')
            
    for folder in os.listdir(TEST_DIR):
        if int(folder.split('-')[-1]) in train_ids:
            os.rename(f'{TEST_DIR}/{folder}', f'{TRAIN_DIR}/{folder}')
        elif int(folder.split('-')[-1]) in val_ids:
            os.rename(f'{TEST_DIR}/{folder}', f'{VAL_DIR}/{folder}')
    
    new_splits = {
        'train': [f'{idx}' for idx in train_ids],
        'val': [f'{idx}' for idx in val_ids],
        'test': [f'{idx}' for idx in test_ids],
    }
    
    with open('./Potholes/splits_out.json', 'w') as f:
        json.dump(new_splits, f, indent=4)