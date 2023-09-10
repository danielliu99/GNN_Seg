import nibabel as nib
import numpy as np
import os 
import argparse
import SimpleITK as sitk

from scripts.preprocess_dataset import swap_labels_to_brats

parser = argparse.ArgumentParser()

parser.add_argument('-r', '--pred_dir', default="", help='path to the directory where data is stored')

parser.add_argument('-v', '--file_list', default="", help='path to the directory where valid list is stored')

parser.add_argument('-s', '--segment', default='_logits.nii.gz', help='mri logits suffix',type=str)

parser.add_argument('-d', '--direction_file', default='data_processing/BraTS-GLI-00005-000-t1c.nii.gz', help='template file for adjust direction',type=str)

parser.add_argument('-t', '--target_folder', help='final output directory',type=str)

parser.add_argument('--logits_label', default='logits')


args = parser.parse_args()

names = [item.split(args.segment)[0] for item in os.listdir(args.pred_dir)]
# with open(args.file_list) as f:
#     for line in f:
#         line = line.strip()
#         name = line.split('/')[-1]
#         names.append(name)

direction_file = os.path.abspath(args.direction_file)
c0 = sitk.ReadImage(direction_file)
Direction = c0.GetDirection()
Origin = c0.GetOrigin()
Spacing = c0.GetSpacing()

for name in names: 
    pred_path = os.path.join(args.pred_dir, name + args.segment) 
    # print(pred_path)
    if args.logits_label == 'logits':
        # print(pred_path.split('/')[-1])
        logits = nib.load(pred_path).get_fdata()
        seg_img = np.argmax(logits, axis=-1)
        seg_img = seg_img.astype(np.float32)
        # seg_img = swap_labels_to_brats(seg_img)
    else: 
        seg_img = nib.load(pred_path).get_fdata()
    
    # ## load direction file
    # # direction_path = os.path.join(args.direction_folder, name, name + '_t1' + args.segment)
    # direction_path = args.direction_file
    # print(direction_path)

    # c0 = sitk.ReadImage(direction_path)
    # Direction = c0.GetDirection()
    # Origin = c0.GetOrigin()
    # Spacing = c0.GetSpacing()
        
    seg_img = sitk.GetImageFromArray(seg_img.transpose(2,1,0))
    seg_img.SetOrigin(Origin)
    seg_img.SetSpacing(Spacing)
    seg_img.SetDirection(Direction)

    save_path = os.path.join(args.target_folder, name.split('_')[-1]+'.nii.gz')
    sitk.WriteImage(seg_img, save_path)
    print(f"Saved successfully: {save_path}")
    print('-'*50)

