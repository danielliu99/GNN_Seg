
import argparse
from numpy import r_, around
from torch.utils.data import Subset
import os


from utils.training_helpers import *
from utils.hyperparam_helpers import populate_hardcoded_hyperparameters, generate_random_hyperparameters
from data_processing.data_loader import ImageGraphDataset
from model.gnn_model import GNN


from numpy.random import randint
import torch
import logging

from tensorboardX import SummaryWriter

'''
Expects a path to a prepared data directory (created by preprocess script)
Can specify either k fold validation or train on full dataset.
Checkpoints model in the specified log folder.
'''

def train_on_full_dataset(args,hyperparams,progress_file_fd,dataset, 
                          load_path = None, 
                          logger = None, 
                          run_time = '', 
                          evaluate = 1, 
                          valid_dataset = None, 
                          writer = None, 
                          last_epoch = 0):
    print("Training on full dataset")
    if load_path: 
            checkpoint = torch.load(load_path)
            cur_epoch = checkpoint['epoch']
    else: 
        checkpoint = None 
        cur_epoch = 1 

    model = GNN(args.model_type,hyperparams,dataset, 
                # load_path=load_path, 
                checkpoint=checkpoint,
                logger=logger, 
                last_epoch=last_epoch, 
                valid_dataset=Subset(valid_dataset, randint(0,len(valid_dataset), 200)))
    if load_path: 
        logger.info(f"Successfully loaded model from {load_path}")


    train_on_fold_verbose(model,args.output_dir+os.sep,
                          hyperparams.n_epochs,args.run_name,1, 
                          progress_file_fd, 
                          logger, 
                          run_time = run_time,
                          writer=writer, 
                          cur_epoch=cur_epoch, 
                          valid_dataset = valid_dataset)
    if evaluate == 1: 
        # train_set_metrics = model.evaluate(Subset(dataset,range(0,len(dataset))))
        train_set_metrics = model.evaluate(Subset(dataset, randint(0, len(dataset), 200)))
        logger.info(f"Evaluate on training set: ")
        logger.info(f"{args.run_name}_full: {train_set_metrics}")

    logger.info(f"Evaluate on validation set: ")
    valid_set_metrics = model.evaluate(Subset(valid_dataset, range(0, len(valid_dataset))))
    logger.info(f"{args.run_name}_full: {valid_set_metrics}")
    document_metrics(progress_file_fd,f"{args.run_name}_full_val",valid_set_metrics)


#if k==1 then train on full dataset
def run_k_fold_val(args,hyperparams,progress_file_fd,dataset,k, 
                load_path = None, 
                logger = None, 
                run_time = '', 
                evaluate = 1):
    assert(k>1)
    folds = chunk_dataset_into_folds(dataset,k)
    for k in range(len(folds)):
        s,e = folds[k]
        val_dataset = Subset(dataset,range(s,e))
        ran = list(r_[0:s,e:len(dataset)])
        train_dataset = Subset(dataset,ran)
        print(f"Fold contains {len(train_dataset)} examples")
        model = GNN(args.model_type,hyperparams,train_dataset, 
                load_path=load_path, 
                logger=logger)
        k=k+1
        checkpoint_dir = args.output_dir+os.sep
        train_on_fold_verbose(model,checkpoint_dir,hyperparams.n_epochs,args.run_name,k, 
                    progress_file_fd, 
                    logger, 
                    run_time = run_time)
        
        if evaluate == 1:
            # train_set_metrics = model.evaluate(train_dataset)
            train_set_metrics = model.evaluate(Subset(train_dataset, randint(0, len(train_dataset), 200)))
            document_metrics(progress_file_fd,f"{args.run_name}_f{k}_train",train_set_metrics)
            
        logger.info(f"Evaluate on validation set: ")
        val_set_metrics = model.evaluate(val_dataset)
        document_metrics(progress_file_fd,f"{args.run_name}_f{k}_val",val_set_metrics)


def document_metrics(fp, description, results):
    counts = results[1]
    metrics = results[0]
    metrics = around(metrics,4)
    print(f"\n#{description} Results#")
    print("Loss:",metrics[0])
    print(f"Predicted Node Counts:", counts[0:4])
    print(f"Label Node Counts:", counts[4:8])
    print(f"WT Node Dice: {metrics[1]}, CT Node Dice: {metrics[2]}, ET Node Dice: {metrics[3]}")
    print(f"WT Voxel Dice: {metrics[4]}, CT Voxel Dice: {metrics[5]}, ET Voxel Dice: {metrics[6]}")
    print(f"WT HD95: {metrics[7]}, CT HD95: {metrics[8]}, ET HD95: {metrics[9]}")
    update_progress_file(fp,description,metrics[0],metrics[4:7])



def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger


if __name__ == '__main__':
    # create the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', help='path to the directory where data is stored',type=str)
    parser.add_argument('-o', '--output_dir', help='Log directory',type=str)
    parser.add_argument('-r', '--run_name', default=None,help='A unique name to save results under',type=str)
    parser.add_argument('-m', '--model_type', default="GSpool",help='What graph learning layer to use. GSpool, GSmean, GSgcn, GAT',type=str)
    parser.add_argument('-k', '--num_folds', default=5, help='How many folds to run k fold validation on. 1== train on full dataset',type=int)
    parser.add_argument('-p', '--data_prefix', default="", help='A prefix that all data folders share, i.e. BraTS2021.',type=str)
    parser.add_argument('-x', '--random_hyperparams', default=False,help='whether to generate random hyperparameters',action='store_true')

    parser.add_argument('-t', '--run_time', default='', help='The n-th time to resume training', type=str)
    parser.add_argument('-e', '--evaluate', default=1, help='Whether to evaluate when training', type=int)
    parser.add_argument('--resume', default=None, help='Path to saved weights', type=str)
    parser.add_argument('--last_epoch', default=0, help='last epoch', type=int)
    parser.add_argument('--valid_dir', default=None, help='path to the directory where validation data is stored',type=str)

    args = parser.parse_args()

    resume = args.resume

    writer = SummaryWriter()

    logger = get_logger(f'{args.output_dir}{os.sep}{args.run_name}_verbose.txt')
    logger.info('-------------------start training!---------------------------')

    dataset = ImageGraphDataset(os.path.expanduser(args.data_dir),args.data_prefix,read_image=False,read_graph=True,read_label=True)

    valid_dataset = ImageGraphDataset(os.path.expanduser(args.valid_dir),args.data_prefix,read_image=False,read_graph=True,read_label=True)

    hyperparams = generate_random_hyperparameters(args.model_type) if args.random_hyperparams else populate_hardcoded_hyperparameters(args.model_type)
    #output dir is where the model weights and progress file are stored, does not output any predictions
    args.output_dir = os.path.expanduser(args.output_dir)
    progress_file_fd = f"{args.output_dir}{os.sep}{args.run_name}.txt"
    create_run_progress_file(progress_file_fd,args.model_type,hyperparams)
    if(args.num_folds==1):
        train_on_full_dataset(args,hyperparams,progress_file_fd,dataset,
                              load_path=resume,
                              logger = logger, 
                              run_time = args.run_time, 
                              evaluate = args.evaluate, 
                              valid_dataset = valid_dataset, 
                              writer = writer, 
                              last_epoch = args.last_epoch)
    elif(args.num_folds>1):
        run_k_fold_val(args,hyperparams,progress_file_fd,dataset,args.num_folds, 
                        load_path=resume,
                        logger = logger, 
                        run_time = args.run_time, 
                        evaluate = args.evaluate)
    else:
        raise ValueError("Number of folds must be a positive integer")

    writer.close()
    logger.info('-------------------FINISHED training!---------------------------')

