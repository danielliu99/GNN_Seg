import os 
import shutil
import random


# Define the paths
Full_raw_folder = "/home/liur1/Documents/brats/data/raw/Training_full"
train_folder = "/home/liur1/Documents/brats/data/raw/train_split/"
test_folder = "/home/liur1/Documents/brats/data/raw/test_split/"


# Get a list of all subfolders in the root folder
subfolders = [f for f in os.listdir(Full_raw_folder) if os.path.isdir(os.path.join(Full_raw_folder, f))]

# Shuffle the list of subfolders randomly
random.shuffle(subfolders)

# Calculate the number of folders for train and test sets
num_train_folders = int(0.8 * len(subfolders))
num_test_folders = len(subfolders) - num_train_folders

# Create train and test folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)


# Copy folders to train and test folders
for i, subfolder in enumerate(subfolders):
    source_path = os.path.join(Full_raw_folder, subfolder)
    if i < num_train_folders:
        destination_path = os.path.join(train_folder, subfolder)
    else:
        destination_path = os.path.join(test_folder, subfolder)
    shutil.copytree(source_path, destination_path)

print(f"{num_train_folders} folders copied to train folder.")
print(f"{num_test_folders} folders copied to test folder.")