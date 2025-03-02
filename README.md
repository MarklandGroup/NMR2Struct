# Accurate and efficient structure elucidation from routine one-dimensional NMR spectra using multitask machine learning

Official implementation of:

**Accurate and efficient structure elucidation from routine one-dimensional NMR spectra using multitask machine learning**

Frank Hu, Michael S. Chen, Grant M. Rotskoff, Matthew W. Kanan, Thomas E. Markland

**Abstract:** Rapid determination of molecular structures can greatly accelerate workflows across many chemical disciplines. However, elucidating structure using only one-dimensional (1D) NMR spectra, the most readily accessible data, remains an extremely challenging problem because of the combinatorial explosion of the number of possible molecules as the number of constituent atoms is increased. Here, we introduce a multitask machine learning framework that predicts the molecular structure (formula and connectivity) of an unknown compound solely based on its 1D <sup>1</sup>H and/or <sup>13</sup>C NMR spectra. First, we show how a transformer architecture can be constructed to efficiently solve the task, traditionally performed by chemists, of assembling large numbers of molecular fragments into molecular structures. Integrating this capability with a convolutional neural network, we build an end-to-end model for predicting structure from spectra that is fast and accurate. We demonstrate the effectiveness of this framework on molecules with up to 19 heavy (non-hydrogen) atoms, a size for which there are trillions of possible structures. Without relying on any prior chemical knowledge such as the molecular formula, we show that our approach predicts the exact molecule 69.6% of the time within the first 15 predictions, reducing the search space by up to 11 orders of magnitude.

Paper available at: https://pubs.acs.org/doi/10.1021/acscentsci.4c01132

Data available on Zenodo at: https://zenodo.org/records/13892026

# Installation
We recommend having [Anaconda](https://www.anaconda.com/download) installed as your package manager and to run this code on a Linux-based GPU cluster. To install NMR2Struct, do the following:
```
git clone https://github.com/MarklandGroup/NMR2Struct.git
cd NMR2Struct
conda env create -f environment.yml
conda activate NMR_env
pip install -e .
```

Make sure to run ``conda update conda`` before installing to ensure your anaconda distribution is up to date. 

# The configuration file

All of the functionality in NMR2Struct is controlled via a YAMl configuration file system. A YAML file consists of sections with fields that can have different user-specified values. While most fields in a YAML file are populated with strings, fields can also be set to booleans (```True``` and ```False```), integers, floats, lists of values, and None types with ```null```. The YAML files in NMR2Struct have the following sections:
- **global_args**: Controls general arguments such as the seed, number of GPUs, where to save everything, and tensor datatypes
- **data**: Determines how input and target data are processed for training and inference, and where to find data files
- **model**: Model architecture arguments that are used for instantiating the model
- **training**: Arguments for the optimization protocol, such as the data split, number of epochs, optimizer, etc.
- **analysis**: Arguments for computing performance metrics on the model 

For the ```data``` section specifically, the ```spectra_file```, ```label_file```, and ```smiles_file``` fields contain lists of files and there should be the same number of elements in each list, even if some elements are set to ```null```. There is an element-wise correspondence of the elements in each list, so the first spectrum, label, and smiles file together forms one dataset, and so on. This is designed so that the training and losses can be computed separately for each dataset and is useful for more advanced early-stopping experiments. The files should have the following formats:
- **spectra**: These should be hdf5 files that contain one dataset called 'spectra' where each row is a one-dimensional vector representing the concatenated <sup>1</sup>H NMR and <sup>13</sup>C NMR spectrum.
- **label**: These should be hdf5 files that contain one dataset called 'substructure_labels' where each row is a one-dimensional binary vector representing the substructures detected in each molecule.
- **smiles**: This should be a numpy file that contains a single array with all SMILES strings encoded into binary format. Be sure to do ```x.decode('utf-8')``` when reading strings from this format.

# Convention for tokens

This section lists how you should set the value for specific alphabet and token related fields within the YAML configuration files. These fields are automatically populated
during training based on the dataset and saved as a completed training YAML config file that you can use. However, if you want to infer a model without training, it is necessary to know what the token value should be. 
This is broken down into two sections, one for the 
substrcture-to-structure transformer and one for the multitask model.

## Substructure-to-structure transformer
| Field | Value |
| -------- | ------- |
| ```model.model_args.source_size``` | number of substructures + 1 (e.g., 958 for the 957 substructures) |
| ```model.model_args.target_size``` | length of alphabet + 3 |
| ```model.model_args.src_pad_token``` | 0 |
| ```model.model_args.tgt_pad_token``` | length of alphabet |
| ```inference.run_inference_args.pred_gen_opts.tgt_start_token``` | length of alphabet + 1 |
| ```inference.run_inference_args.pred_gen_opts.tgt_stop_token``` | length of alphabet + 2 |

## Spectrum-to-structure multitask model
| Field | Value |
| -------- | ------- |
| ```model.model_args.structure_model_args.source_size``` | 28045 |
| ```model.model_args.structure_model_args.target_size``` | length of alphabet + 3|
| ```model.model_args.structure_model_args.src_pad_token``` | null | 
| ```model.model_args.structure_model_args.tgt_pad_token``` | length of alphabet |
| ```model.model_args.substructure_model_args.source_size``` | 28045 | 
| ```model.model_args.substructure_model_args.src_pad_token``` | null |
| ```inference.run_inference_args.pred_gen_opts.tgt_start_token``` | length of alphabet + 1 |
| ```inference.run_inference_args.pred_gen_opts.tgt_stop_token``` | length of alphabet + 2 |

# Training 
To train a substructure-to-structure transformer, follow these steps:
1. Modify the training configuration file under ```example_configs/training_substruct_to_struct.yaml``` to point to the correct substructure file at ```data.label_file``` and the correct smiles file under the ```data.smiles_file```. The ```data.spectra_file``` field can be set to a list of ```null``` values because spectra are not used for training the substructure-to-structure model.
2. Modify the config file to point to the correct splitting file at  ```training.splits```. The splitting file should be a dictionary of three fields which specify the indices that should go into the training, validation, and testing set, respectively. If you do not want to provide a splitting dictionary, set this field to ```null``` in which case the 
splitting is done based on the ```train_size```, ```val_size```, and ```test_size``` fractions.
3. Copy the training config file into a directory where you want to run the training.
4. Run the following command (assuming you renamed your YAML file to be ```config.yaml```):
```
nmr_train config.yaml
```

A similar procedure is used for training a spectrum-to-structure multitask model, but make sure to:
- Use the ```example_configs/training_spectrum_to_struct.yaml``` file as the starting point and set the ```spectra_file``` field to list the correct spectra files.
- Set the ```model.model_args.structure_model_ckpt``` to the correct checkpoint if using a pretrained transformer, otherwise leave ```null```.
- Set the ```data.alphabet``` as either ```null``` if not using a pretrained transformer or to the alphabet used by the pretrained transformer.

# Inference (with full dataset)
> [!WARNING]
> This inference workflow is intended for running inference on a model on a dataset that was used for training, i.e. getting predictions over a test set to then compute metrics on. If you want to infer a single example, please refer to the section on inference with specific examples.

To infer a substructure-to-structure transformer for SMILES generation, first make sure to set the alphabet correctly with:
```yaml
inference:
  ...
  run_inference_args: 
    ...
    pred_gen_opts:
      ...
      alphabet: checkpoints/alphabet.npy
```
which is the alphabet generated from training. 

> [!NOTE]
> If you want to use the alphabet that was used in the original model from the paper, put the path to the file ```example_configs/alphabet.npy``` instead.

Once the alphabet is set, just run:
```
nmr_infer config.yaml 0 1
```
where ```config.yaml``` is your modified training script from the training section which should contain an inference section tailored for the substructure-to-structure transformer. The 0 and 1 refer to the local rank and total number of GPU processes and is used to control multi-GPU inference. For most cases, inference on one GPU is sufficient, so we use a local rank of 0 and 1 GPU process. Make sure to set the ```inference.splits``` field correctly. The same procedure holds for inferring SMILES from a spectrum-to-structure multitask model. By default, the ```inference``` sections of the provided YAML files are configured to have the models generate SMILES strings. 

To infer substructures from the multitask model, modify the inference section as follows, being sure to fill in the splits file which denotes which parts of the dataset belong to the test set:
```yaml
inference:
  model_selection: lowest
  splits: 
    - PATH TO SPLITS FILE OR NULL
  train_size: 0.8 
  val_size: 0.1 
  test_size: 0.1 
  dloader_args:
    shuffle: false
    batch_size: 1024
  sets_to_run: ['test']
  run_inference_args: 
    pred_gen_fn: 'infer_multitask_model_substructures'
    pred_gen_opts:
      track_gradients: true
    write_freq: 100
```
Running inference writes the raw predictions as an hdf5 file to the save directory that you specified under the ```global_args``` section, which should also be where the model checkpoints and tensorboard files are saved.


# Inference (on a specific spectrum)
To infer the structures and/or substructures from a specific set of spectra, we recommend using the single spectrum inference entry point. First, be sure to set the ```inference.run_inference_args.pred_gen_opts.tgt_start_token``` and ```inference.run_inference_args.pred_gen_opts.tgt_stop_token``` correctly according to the section on token conventions.
To use this entry point, do the following:
```
nmr_infer_single_spectrum \
    --config YAML_CONFIG_FILE \
    --hnmr_file HNMR_FILE \
    --cnmr_file CNMR_FILE \
    --hnmr_shifts HNMR_SHIFTS \
    --cnmr_shifts CNMR_SHIFTS \
    --ckpt CKPT \
    --normalize
```
> [!NOTE]
> This entry point only works for inferring structure/substructures from spectra using a multitask model.

The arguments are as follows: 
- ```--config```: A YAML configuration file that contains at least the ```global_args```, ```model```, and ```inference``` sections. The ```inference``` section should be configured 
correctly for SMILES or substructure inference. You can copy the correct inference settings from the YAML configuration files in ```example_configs```.
- ```--hnmr_file```: A text or tab separated csv file containing pairs of (ppm, intensity) values on each row separated by a space. If not using an <sup>1</sup>H NMR spectrum, pass "None" to this argument.
- ```--cnmr_file```: A text file containing a comma separated list of the carbon ppm shifts in the NMR. If not using an <sup>13</sup>C NMR spectrum, pass "None" to this argument.
- ```--hnmr_shifts```: A pickle file containing the shift grid as a 1D numpy array that is used to interpolate the <sup>1</sup>H NMR spectrum. You can find the shift grid used in the paper in ```example_configs/HNMR_shifts.p```.
- ```--cnmr_shifts```: A pickle file containing the shift grid for discretizing the <sup>13</sup>C NMR spectrum. The grid used in the paper can be found in ```example_configs/CNMR_shifts.p```.
- ```--ckpt```: The model checkpoint you want to use. Its architecture should match the one specified in ```--config```.
- ```---normalize```: Toggles normalization of the <sup>1</sup>H NMR spectrum by dividing the spectrum by the its highest intensity.  

At the end of the inference, the predictions will be saved to the ```global_args.savedir``` as an hdf5 file under the ```test``` group. You must pass in at least one spectrum, either an <sup>1</sup>H or <sup>13</sup>C NMR, to the respective argument. 

# Analysis
> [!WARNING]
> Analysis is only useful when you are training a model on a dataset with known targets and you want to compare its predictions against the target values. If you are just doing a one-off inference using the model, analysis will not work. 

If you want to calculate metrics for substructures, configure the ```analysis``` section of your YAML file as follows:
```yaml
analysis:
  analysis_type: "substructure"
  pattern: "predictions_dataset_0_[0-9]+.h5"
```
This will calculate metrics such as RMSEs, F1-scores, BCE values, etc. and saves the results as a dictionary pickle file at ```global_args.savedir```. The format of the dictionary is as follows:

```
substructure_metrics.pkl
├── test
│   ├── substruct_losses
│   ├── substruct_avg_loss
│   ├── substruct_accs
│   ├── substruct_avg_acc
│   ├── roc_auc_score
│   ├── precision
│   ├── recall
│   ├── prc_auc_score
│   ├── fscore
│   ├── exact_seq_match_percent
│   ├── avg_seq_bce
│   ├── root_power_2
│   ├── root_power_3
│   └── root_power_4
└── ...
```

Where the `...` denotes additional keys corresponding to the training and validation sets if analysis was done on them. For comparison with the metrics presented in the paper, the value that we care about is the 
substructure F1 score, which corresponds to the key `fscore` in the dictionary.

For SMILES analysis, configure the ```analysis``` section as follows:
```yaml
analysis:
  analysis_type: "SMILES"
  pattern: "predictions_dataset_0_[0-9]+.h5"
  f_addn_args: 
    substructures: PATH TO SUBSTRUCTURE FILE
```
where the substructures is set to point to the ```example_configs/substructures_957.p``` file which contains the SMART strings for the 957 substructures used. The results will be saved to 
a hdf5 file called ```processed_predictions.h5``` located at ```global_args.savedir```. The structure of ```processed_predictions.h5``` is as follows:

```
processed_predictions.h5
├── test
│   ├── bad_predictions
│   │   ├── predictions
│   │   ├── scores
│   │   └── targets
│   └── valid_predictions
│       ├── 0
│       │   ├── num_heavy_atoms
│       │   ├── prediction_bce_losses
│       │   ├── prediction_scores
│       │   ├── prediction_strings
│       │   └── target
│       └── ...
└── ...
```
For each subset of the data (```train```, ```val```, ```test```) there is a ```bad_predictions``` and a ```valid_predictions``` key. The ```bad_predictions``` key contains the set of all cases where the model was unable to generate
any valid SMILES strings within the specified number of predictions for each target. The field ```bad_predictions.predictions``` contains an array of shape (`num_failed`, `num_pred_per_tgt`) where each row contains all the generated predictions 
that were unparasable for a given target. The targets themselves are scored in the ```bad_predictions.targets``` key. The ```bad_predictions.scores``` key contains the sum of the log probabilities for the given sequence, so the exponential of this 
quantity can be considered as the total probability of the given sequence to be sampled from the transformer. 

The ```valid_predictions``` key contains the set of all targets for which the model was able to predict at least one valid SMILES strings. Each entry is designated by zero-indexed number string. Within the group for each molecule there are five keys:

- ```num_heavy_atoms```: The number of heavy (non-hydrogen) atoms in the given molecule.
- ```prediction_bce_losses```: The binary cross entropy losses computed between the target molecule and each of the predictions by representing all molecules in terms of their substructure vectors. The lower the loss the better. For numerical reasons, values on the order of 1e-16 should be considered zero.
- ```prediction_scores```: The score for each prediction, which is again the sum of log probabilities computed along the sequence.
- ```prediction_strings```: The predicted and canonicalized SMILES strings.
- ```target```: The SMILES string target that the model is trying to predict.

Calculation of any performance metrics should be done over the ```processed_predictions.h5``` file, and it should be very straightforward to compute things such as string prediction accuracy by simply checking if for each valid prediction the target appears within the set of predictions, which is indeed
how the metrics were computed in the paper.

# Tensorboard visualization
NMR2Struct uses [Tensorboard](https://www.tensorflow.org/tensorboard) to visualize the learning curves when training the models, and it is installed with the environment. To use Tensorboard, do the following:
```
cd /path/to/training/directory
tensorboard --logdir=./
```
This should automatically do port-forwarding to your browser where you will see the Tensorboard dashboard. Tensorboard searches for all event files in a parent directory, so you can 
visualize multiple runs at once if they are all contained within the same parent directory.
