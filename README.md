This repository is for brain tumor segmentation using Graph Neural Network. GraphSAGE is used as the backbone framework. The data set is provided by the International Brain Tumor Segmentation (BraTS) challenge. The training set contains 5880 MRI scans in 4 modalities from 1,470 brain diffuse glioma patients. 

## Use this repo
### Preprocessing

```bash
python -m scripts.preprocess_dataset -d <path_to_raw_data> -n 20000 -k 0 -b 0.5 -o <path_to_processed_data>
```

### Training 

```bash 
python -m scripts.train_gnn -d <path_to_training_data> --valid_dir <path_to_validating_data> -o ./logs -k 1 -m GSpool
```

### Inferencing 

```bash 
python -m scripts.generate_gnn_predictions -d <path_to_testing_data> -o <path_to_prediction> -w <path_to_trained_weights> -f logits 
```

### Postprocessing to BRATS format 

```bash 
python -m scripts.logits_to_preds -r <path_to_prediction> -t <path_to_output>
```

## Use trained model to make prediction
You can pull the docker images to make predictions directly. 

```bash 
sudo docker pull lrshow99/gnn_seg

sudo docker run -it --rm --gpus all -v <path_to_testing_data>:/test_folder -v <path_to_save_prediction>:/out_folder gnn_seg:latest 
```




## Citation 
[1] U.Baid, et al., The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification, arXiv:2107.02314, 2021.

[2] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[3] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

In addition, if there are no restrictions imposed from the journal/conference you submit your paper about citing "Data Citations", please be specific and also cite the following:

[4] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q

[5] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.GJQ7R0EF

[6] Saueressig, C., Berkley, A., Munbodh, R., Singh, R. (2022). A Joint Graph and Image Convolution Network for Automatic Brain Tumor Segmentation. In: Crimi, A., Bakas, S. (eds) Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. BrainLes 2021. Lecture Notes in Computer Science, vol 12962. Springer, Cham. https://doi.org/10.1007/978-3-031-08999-2_30