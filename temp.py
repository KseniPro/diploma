import os
import glob

# Define the directories containing the images and JSON files
image_dir = 'lps/images'
json_dir = 'lps/labels_for_faster-rcnn'

# Get a list of all image files in the image directory
image_files = glob.glob(os.path.join(image_dir, '*'))
image_files.sort()  # Sort the files to maintain a consistent order

# Initialize a counter for the filenames
counter = 1

# Iterate over each image file
for image_file in image_files:
    # Extract the basename without the extension
    basename = os.path.splitext(os.path.basename(image_file))[0]
    
    # Construct the new filename based on the counter
    new_name = f"LPS_{counter}"
    
    # Construct the full new file paths for the image and JSON files
    new_image_path = os.path.join(image_dir, f"{new_name}{os.path.splitext(image_file)[1]}")
    new_json_path = os.path.join(json_dir, f"{new_name}.json")
    
    # Rename the image file
    os.rename(image_file, new_image_path)
    
    # Construct the path to the corresponding JSON file
    json_file = os.path.join(json_dir, f"{basename}.json")
    
    # Check if the corresponding JSON file exists
    if os.path.exists(json_file):
        # Rename the JSON file
        os.rename(json_file, new_json_path)
    
    # Increment the counter
    counter += 1