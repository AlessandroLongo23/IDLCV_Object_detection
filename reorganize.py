import os
import json
import shutil

def reorganize_in_folders():
    # Set the path to the folder containing the files
    source_folder = "Potholes/annotated-images"

    # Loop through the files in the folder
    for filename in os.listdir(source_folder):
        # Check if the file has a matching .xml file
        if filename.endswith(".jpg"):
            base_name = os.path.splitext(filename)[0]
            xml_file = f"{base_name}.xml"
            
            # Check if the XML file exists
            if os.path.isfile(os.path.join(source_folder, xml_file)):
                # Create the subfolder
                subfolder = os.path.join(source_folder, base_name)
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder)
                
                # Move the files to the subfolder
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

    # train
    train_folder = os.path.join(source_folder, 'train')
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    for folder in [split.split('.')[0] for split in splits['train']]:
        folder_path = os.path.join(source_folder, folder)
        shutil.move(folder_path, os.path.join(train_folder))
        
    # test
    test_folder = os.path.join(source_folder, 'test')
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    for folder in [split.split('.')[0] for split in splits['test']]:
        folder_path = os.path.join(source_folder, folder)
        shutil.move(folder_path, os.path.join(test_folder))
        
def delete_data_json():
    for folder in os.listdir('./Potholes/annotated-images/test'):
        if os.path.exists(f'./Potholes/annotated-images/test/{folder}/data.json'):
            os.remove(f'./Potholes/annotated-images/test/{folder}/data.json')
            
def delete_proposals_folder():
    for folder in os.listdir('./Potholes/annotated-images/train'):
        if os.path.exists(f'./Potholes/annotated-images/train/{folder}/proposals'):
            shutil.rmtree(f'./Potholes/annotated-images/train/{folder}/proposals')
            
def rename_data_to_ground_truth():
    for split in ['train', 'test']:
        for folder in os.listdir(f'./Potholes/annotated-images/{split}'):
            os.rename(f'./Potholes/annotated-images/{split}/{folder}/data.xml', f'./Potholes/annotated-images/{split}/{folder}/ground_truth.xml')
            
rename_data_to_ground_truth()