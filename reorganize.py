import os
import shutil

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