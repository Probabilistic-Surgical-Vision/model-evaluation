# Depth-Uncertainty Model Evaluation Package

This repository holds the code for evaluating a depth-uncertainty model on the Hamlyn and SCARED Datasets. The package assumes the model output is similar to Monodepth2, where the first two channels are the left and right disparity, and the last two are the model's prediction of the left and right uncertainty.


## Acknowledgements

The SCARED video converter package was adapted from [llreda/Stereo_Matching](https://github.com/llreda/Stereo_Matching/blob/master/processing/dataset_processing.py). This is what is used to

- Convert the videos to stereo pairs.
- Rectify the images, depth maps and keyframes.
- Calculate the focal length and baseline, in order to calculate depth from disparity during evaluation.

The evaluation methods chosen were inspired by Tukra et al. in [_Randomly-connected neural networks for self-supervised depth estimation_](https://www.tandfonline.com/doi/full/10.1080/21681163.2021.1997648).

## Pre-requisites and installation

To use this package, you will need Python 3.6 or higher. Using an NVIDIA GPU, such as an RTX6000 is recommended.

Download the repository from GitHub and create a virtual environment and activate it:
```bash
python -m venv venv
. venv/bin/activate
```

Install all the packages from pip
```bash
python -m pip install -r requirements.txt
```

## Usage

To use the evaluation package, store the models to be tested in a folder called `models`. Create a subfolder for each dataset (i.e. `da-vinci` and `scared`). Make sure to give the model the same basename in each folder.

Run the notebook, check the config file used is compatible with the model (for the aleatoric and epistemic models, use the `uncertainty-config.yml` file and for the model proposed by Tukra et al., use `original-config.yml`).

Re-run the notebook with different models, using the `model_name` variable to set which one is loaded for evaluation.