import os

# Get the current directory
data_dir = "/home/liur1/Documents/brats/data/processed/test_split"

# Get a list of all folder names in the current directory
folder_names = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

# Specify the name of the text file where you want to write the folder names
output_file = "subjects.txt"
output_path = os.path.join(data_dir, output_file)

# Write the folder names to the text file, one per line
with open(output_path, "w") as f:
    for folder_name in folder_names:
        f.write(folder_name + "\n")

print(f"Subject list written to {output_path}")
