import numpy as np
import pickle, os, shutil
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional
from torch.utils.tensorboard import SummaryWriter

def train_loop(model: nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: Callable[[Tensor, Tensor], Tensor], 
               optimizer: torch.optim.Optimizer, 
               epoch: int, 
               writer: torch.utils.tensorboard.SummaryWriter, 
               scheduler: Optional[torch.optim.lr_scheduler.LambdaLR], 
               write_freq: int = 100) -> float:
    """Model training loop
    Args:
        model: The model to train
        dataloader: The dataloader for the training dataset
        loss_fn: The loss function to use for the model, with the signature
            tensor, tensor -> tensor
        optimizer: The optimizer for training the model
        epoch: The current epoch
        writer: Tensorboard writer for logging losses and learning rates
        scheduler: The optional learning rate scheduler
        write_freq: The frequency for printing loss information
    """
    tot_loss = 0
    model.train()
    for ibatch, (x, y) in enumerate(dataloader):
        inner_step = int(( epoch * len(dataloader)) + ibatch)
        loss = model.get_loss(x,y,loss_fn) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Step the learning rate scheduler too based on the current optimizer step
        if scheduler is not None:
            scheduler.step()
        if (ibatch % write_freq == 0):
            print(f"Epoch: {epoch}\tBatch:{ibatch}\tTrain Loss:{loss.item()}")
        tot_loss += loss.item()
        writer.add_scalar("Training Step Loss", loss.item(), inner_step)
    writer.add_scalar("Avg. Epoch Train Loss", tot_loss / len(dataloader), epoch)
    return tot_loss / len(dataloader)

def validation_loop(model: nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    loss_fn: Callable[[Tensor, Tensor], Tensor], 
                    epoch: int, 
                    writer: torch.utils.tensorboard.SummaryWriter, 
                    write_freq: int = 100) -> float:
    """Model validation loop
    Args:
        model: The model to validate
        dataloader: The dataloader for the validation dataset
        loss_fn: The loss function to use for the model, with the signature
            tensor, tensor -> tensor
        epoch: The current epoch
        writer: Tensorboard writer for logging losses and learning rates
        write_freq: The frequency for printing loss information
    """
    tot_loss = 0
    model.eval()
    with torch.no_grad():
        for ibatch, (x, y) in enumerate(dataloader):
            inner_step = int(( epoch * len(dataloader)) + ibatch)
            loss = model.get_loss(x,y,loss_fn) 
            if (ibatch % write_freq == 0):
                print(f"Epoch: {epoch}\tBatch:{ibatch}\tValidation Loss:{loss.item()}")
            tot_loss += loss.item()
            writer.add_scalar("Validation Step Loss", loss.item(), inner_step)
        writer.add_scalar("Avg. Epoch Validation Loss", tot_loss / len(dataloader), epoch)
    return tot_loss / len(dataloader)

def test_loop(model: nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: Callable[[Tensor, Tensor], Tensor], 
                epoch: int, 
                writer: torch.utils.tensorboard.SummaryWriter, 
                write_freq: int = 100) -> float:
    """Model test loop
    Args:
        model: The model to test
        dataloader: The dataloader for the test dataset
        loss_fn: The loss function to use for the model, with the signature
            tensor, tensor -> tensor
        epoch: The current epoch
        writer: Tensorboard writer for logging losses and learning rates
        write_freq: The frequency for printing loss information
    """
    tot_loss = 0
    model.eval()
    with torch.no_grad():
        for ibatch, (x, y) in enumerate(dataloader):
            inner_step = int(( epoch * len(dataloader)) + ibatch)
            loss = model.get_loss(x,y,loss_fn) 
            if (ibatch % write_freq == 0):
                print(f"Epoch: {epoch}\tBatch:{ibatch}\Test Loss:{loss.item()}")
            tot_loss += loss.item()
            writer.add_scalar("Test Step Loss", loss.item(), inner_step)
        writer.add_scalar("Avg. Epoch Test Loss", tot_loss / len(dataloader), epoch)
    return tot_loss / len(dataloader)

def save_model(model: nn.Module, 
               optim: torch.optim.Optimizer, 
               epoch: int, 
               loss_metric: float, 
               savedir: str, 
               tag: str = "") -> str:
    """Save model and optimizer state dicts to file
    Args:
        model: The model to save
        optim: The optimizer to save
        epoch: The current epoch
        loss_metric: The loss value to associate with the 
        savedir: The directory to save the model and optimizer state dicts
        tag: An optional tag to add to the model name
    """
    if tag == "":
        savename = f'{savedir}/model_epoch={epoch}_loss={loss_metric:.8f}.pt'
    else:
        savename = f'{savedir}/model_epoch={epoch}_loss={loss_metric:.8f}_{tag}.pt'
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict()},
                savename)
    return savename
    
def delete_checkpoint(checkpoint: str) -> None:
    """Delete a checkpoint file
    Args:
        checkpoint: The path to the checkpoint file
    """
    os.remove(checkpoint)

def fit(model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader, 
        val_dataloader: torch.utils.data.DataLoader, 
        test_dataloader: torch.utils.data.DataLoader, 
        loss_fn: Callable[[Tensor, Tensor], Tensor], 
        optimizer: torch.optim.Optimizer, 
        nepochs: int,
        save_dir: str,
        writer: torch.utils.tensorboard.SummaryWriter, 
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR], 
        top_checkpoints_n: int = 10,
        loss_metric: str = 'val',
        write_freq: int = 100,
        test_freq: int = 10,
        prev_epochs: int = 0) -> tuple[list, list, list, list]:
    """Model training loop

    Args:
        model: The model to train
        train_dataloader: The dataloader for the training dataset
        val_dataloader: The dataloader for the validation dataset
        test_dataloader: The dataloader for the test dataset
        loss_fn: The loss function to use for the model, with the signature
            tensor, tensor -> tensor
        optimizer: The optimizer for training the model
        nepochs: The number of epochs to train for
        save_dir: The directory to save the model and optimizer state dicts
        writer: Tensorboard writer for logging losses and learning rates
        scheduler: The optional learning rate scheduler
        top_checkpoints_n: The number of top checkpoints to save
        loss_metric: The criterion to use for saving checkpoints. Can be 'val' or 'train'
        write_freq: The frequency for printing loss information
        test_freq: The frequency for running the test loop of the model
        prev_epochs: The number of epochs that have already been trained for. This is used
            for loading checkpoints
    """
    best_losses = np.ones(top_checkpoints_n) * np.inf
    model_names = [None] * top_checkpoints_n
    train_losses, val_losses, test_losses = [], [], []
    for epoch in range(nepochs):
        true_epoch = epoch + prev_epochs
        train_loss = train_loop(model, 
                                train_dataloader, 
                                loss_fn, 
                                optimizer, 
                                true_epoch, 
                                writer, 
                                scheduler, 
                                write_freq)
        train_losses.append(train_loss)
        val_loss = validation_loop(model, 
                                   val_dataloader, 
                                   loss_fn, 
                                   true_epoch, 
                                   writer, 
                                   write_freq)
        val_losses.append(val_loss)
        if true_epoch % test_freq == 0:
            test_loss = test_loop(model,
                                test_dataloader,
                                loss_fn,
                                true_epoch,
                                writer,
                                write_freq)
            test_losses.append(test_loss)
        
        curr_k_metric_value = train_loss if loss_metric == 'train' else val_loss
        max_loss_idx = np.argmax(best_losses)
        max_loss_value = best_losses[max_loss_idx]
        max_loss_model = model_names[max_loss_idx]
        if curr_k_metric_value < max_loss_value:
            if max_loss_model is not None:
                delete_checkpoint(max_loss_model)
            model_name = save_model(model, 
                                    optimizer, 
                                    true_epoch, 
                                    curr_k_metric_value, 
                                    save_dir)
            #Update loss value and model names
            best_losses[max_loss_idx] = curr_k_metric_value
            model_names[max_loss_idx] = model_name
    writer.flush()
    writer.close()
    final_test_loss = test_loop(model,
                                test_dataloader,
                                loss_fn,
                                true_epoch,
                                writer,
                                write_freq)
    test_losses.append(final_test_loss)
    if nepochs >= top_checkpoints_n:
        assert(None not in model_names)
        assert(all(best_losses < np.inf))
    return train_losses, val_losses, test_losses, model_names