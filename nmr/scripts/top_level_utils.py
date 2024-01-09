import numpy as np
import random
import torch
import yaml
from typing import Union, Optional
from torch.utils.data import Dataset
import os
import h5py
import pickle as pkl
from functools import reduce

def seed_everything(seed: Union[int, None]) -> int:
    if seed is None:
        seed = random.randint(0, 100000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker() -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def dtype_convert(dtype: str) -> torch.dtype:
    '''Maps string to torch dtype'''
    dtype_dict = {
        'float32': torch.float,
        'float64': torch.double,
        'float16': torch.half
    }
    return dtype_dict[dtype]

def save_completed_config(config: dict, savedir: str) -> None:
    '''Saves the completed config file to the savedir'''
    with open(f"{savedir}/full_config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def split_data_subsets(dataset: Dataset,
                       splits: Optional[str],
                       train_size: float = 0.8,
                       val_size: float = 0.1,
                       test_size: float = 0.1) -> tuple[Dataset, Dataset, Dataset]:
    '''Splits the dataset using indices from passed file
    Args:
        dataset: The dataset to split
        splits: The path to the numpy file with the indices for the splits
        train_size: The fraction of the dataset to use for training
        val_size: The fraction of the dataset to use for validation
        test_size: The fraction of the dataset to use for testing
    '''
    if splits is not None:
        print(f"Splitting data using indices from {splits}")
        split_indices = np.load(splits, allow_pickle = True)
        train, val, test = split_indices['train'], split_indices['val'], split_indices['test']
        return torch.utils.data.Subset(dataset, train), torch.utils.data.Subset(dataset, val), \
            torch.utils.data.Subset(dataset, test)
    else:
        assert(train_size + val_size + test_size == 1)
        print(f"Splitting data using {train_size} train, {val_size} val, {test_size} test")
        train, val, test = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        return train, val, test
    
def save_train_history(savedir: str, loss_obj: tuple[list, ...]) -> None:
    """Saves the training history of the model to the specified directory
    
    Args:
        savedir: The directory to save the training history
        loss_obj: The tuple of lists containing the training history, with
            train_losses
            val_losses
            test_losses
            model_names
            best_losses
        in that order
    """
    train_losses, val_losses, test_losses, model_names, best_losses = loss_obj
    print("Saving losses")
    with h5py.File(f"{savedir}/losses.h5", "w") as f:
        f.create_dataset("train_losses", data = train_losses)
        f.create_dataset("val_losses", data = val_losses)
        f.create_dataset("test_losses", data = test_losses)
    
    with open(f"{savedir}/model_names_losses.pkl", "wb") as f:
        pkl.dump((model_names, best_losses), f)

def save_str_set(h5ptr: h5py.File, 
                 preds: list[tuple],
                 savename: str) -> None:
    """Saves string predictions to the passed h5 file pointer

    Args:
        h5ptr: The h5 file pointer to save to
        preds: The list of predictions, tuples of form (tgt, [pred, ...])
        savename: The name to save the predictions under

    In h5py, python strings are automatically encoded as variable-length UTF-8.
    It is assumed that each target has the same number of generated predictions
    """
    targets = [elem[0] for elem in preds]
    predictions = [elem[1] for elem in preds]
    group = h5ptr.create_group(savename)
    group.create_dataset("targets", data = targets)
    group.create_dataset("predictions", data = predictions)

def find_max_length(preds: list[list[np.ndarray]]) -> int:
    flattened_preds = list(reduce(lambda x, y : x + y, preds))
    max_len = max([len(elem) for elem in flattened_preds])
    return max_len

def pad_single_prediction(pred: list[np.ndarray],
                          max_len: int, 
                          pad_token: int) -> np.ndarray:
    fixed_seqs = []
    for elem in pred: 
        padded_elem = np.pad(elem, 
                             (0, max_len - len(elem)), 
                             'constant', 
                             constant_values = (pad_token,))
        fixed_seqs.append(padded_elem)
    return np.array(fixed_seqs)

def save_array_set(h5ptr: h5py.File, 
                   preds: list[tuple],
                   savename: str) -> None:
    """Saves array predictions to the passed h5 file pointer

    Args:
        h5ptr: The h5 file pointer to save to
        preds: The list of predictions, tuples of form (tgt, [pred, ...])
        savename: The name to save the predictions under

    padding is done on these arrays to ensure size consistency when saving to hdf5
    """
    targets = [elem[0] for elem in preds]
    targets = np.array(targets)
    predictions = [elem[1] for elem in preds]
    max_len = find_max_length(predictions)
    pad_token = 999_999
    predictions = [pad_single_prediction(elem, max_len, pad_token) for elem in predictions]
    predictions = np.array(predictions)
    group = h5ptr.create_group(savename)
    group.create_dataset("targets", data = targets)
    group.create_dataset("predictions", data = predictions)
    group.create_dataset("additional_pad_token", data = pad_token)

def save_inference_predictions(savedir: str,
                               train_predictions: list,
                               val_predictions: list,
                               test_predictions: list) -> None:
    '''Saves the predictions from inference as h5 files in the specified directory

    Args:
        savedir: The directory to save to
        train_predictions: The list of train predictions
        val_predictions: The list of validation predictions
        test_predictions: The list of test predictions

    The formatting for the h5py file changes depending on the form of the predictions. 
    However, at the top level, the following are exposed: 'train', 'val', and 'test'. Depending
    on if predictions are passed for each set, each key further has the two subkeys 'target' and
    'predictions'.
    .
    '''    
    print("Saving predictions...")
    test_element = train_predictions[0]
    assert(isinstance(test_element, tuple))
    with open(f"{savedir}/predictions.h5", 'w') as f:    
        if isinstance(test_element[0], str):
            save_fxn = save_str_set
        elif isinstance(test_element[0], np.ndarray):   
            save_fxn = save_array_set
        if train_predictions is not None:
            save_fxn(f, train_predictions, 'train')
        if val_predictions is not None:
            save_fxn(f, val_predictions, 'val')
        if test_predictions is not None:
            save_fxn(f, test_predictions, 'test')

def save_token_size_dict(savedir: str,
                         token_size_dict: dict) -> None:
    with open(f"{savedir}/token_size_dict.pkl", "wb") as f:
        pkl.dump(token_size_dict, f)