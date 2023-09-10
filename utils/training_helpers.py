import torch 
import os 
from torch.utils.data import Subset


'''
A collection of helper functions used while training a model.
'''

#save hyperparameters to a log file at a specified path
def create_run_progress_file(fp,model_type,hp):
    with open(fp, "w") as f:
        f.write(f"----Model Parameters----\n")
        f.write(f"Model\t{model_type}\n")
        f.write(f"Epochs\t{hp.n_epochs}\n")
        f.write(f"Input Features\t{hp.in_feats}\n")
        f.write(f"LR\t{hp.lr}\n")
        f.write(f"L2Reg\t{hp.w_decay}\n")
        f.write(f"LR Decay\t{hp.lr_decay}\n")
        f.write(f"Layer Sizes\t{hp.layer_sizes}\n")

        #GAT only
        if(model_type=="GAT"):
            f.write(f"Heads\t{hp.gat_heads}\n")
            f.write(f"Residuals\t{hp.gat_residuals}\n")

        f.write("Fold\tLoss\tWT_Dice\tCT_Dice\tET_Dice\n\n")


def chunk_dataset_into_folds(dataset,k):
    fold_size = len(dataset)//k
    folds = []
    for i in range(k):
        folds.append((i*fold_size,(i+1)*fold_size))
    return folds


def update_progress_file(fp,description,loss,dices):
    with open(fp, "a") as f:
        f.write(f"{description}\t{loss}\t{dices[0]}\t{dices[1]}\t{dices[2]}\n")

#Pass in a model (which already contains the training data) and run it for a specified number of epochs.
#The model checkpoints its weights every couple of epochs.
def train_on_fold(model,checkpoint_dir,n_epoch,run_name,fold):
    lowest_loss=1000
    for i in range(1,n_epoch+1):
        epoch_loss=model.run_epoch()
        #TODO: revert
        if(i%1==0):
            print(f"____Epoch {i}_____")
            print(epoch_loss)
            if(i>n_epoch/2 and epoch_loss>lowest_loss+0.001):
                print("Fold terminated early due to converged train loss")
                print(f"Ran for {i} epochs")
                return
            if epoch_loss<lowest_loss:
                lowest_loss=epoch_loss
                #checkpoint fold
                description = f"{run_name}_f{fold}"
                model.save_weights(checkpoint_dir,description)
    print(f"Finished fold {fold} for run {run_name}")


def train_on_fold_verbose(model,checkpoint_dir,n_epoch,run_name,fold, 
                          progress_file_fd, 
                          logger, 
                          run_time = '', 
                          writer = None, 
                          cur_epoch = 1, 
                          valid_dataset = None):
    lowest_loss=1000
    logger.info(f"Model is running on {model.device} \n")

    # for i in range(1,n_epoch+1):
    for i in range(cur_epoch, n_epoch+1):
        # epoch_loss=model.run_epoch()
        epoch_loss, optimizer, lr_decay = model.run_epoch()
        if writer:
            writer.add_scalar('lr', epoch_loss, i)
        #TODO: revert
        if(i%1==0):
            logger.info(f">>>>>>>>Epoch{i} finished")
            # logger.info(f">>>>>>>>Epoch{i} / {n_epoch} \n {epoch_loss} \n ")
            if(i>n_epoch/2 and epoch_loss>lowest_loss+0.001):
                print("Fold terminated early due to converged train loss")
                print(f"Ran for {i} epochs")
                return
            if epoch_loss<lowest_loss or epoch_loss < 0.1:
                lowest_loss=epoch_loss
                #checkpoint fold
                if i%5 == 0:
                    # save current model
                    description = f"{run_name}_f{fold}_r{run_time}"
                    # model.save_weights(checkpoint_dir,description)
                    checkpoint = {
                        "model": model.net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        "epoch": i,
                        'lr_decay': lr_decay.state_dict()
                    }
                    torch.save(checkpoint, os.path.join(checkpoint_dir, description))
                    logger.info(f"Model: {description} saved successfully \n")
                    if valid_dataset: 
                        valid_set_metrics = model.evaluate(Subset(valid_dataset, range(0, len(valid_dataset), 5)))
                        print("Loss:",valid_set_metrics[0])
                        if writer:
                            writer.add_scalar('evaluate_error', valid_set_metrics[0][0], i)
                    # evaluate 
                    # train_set_metrics = model.evaluate(model.valid_dataset)
                    # logger.info(f"Evaluate on training set: ")
                    # logger.info(f"{run_name}_full: {train_set_metrics}")
                   

    logger.info(f"Finished fold {fold} for run {run_name}")
    # print(f"Finished fold {fold} for run {run_name}")