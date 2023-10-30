# pl-experiments-template
A very simple template for PyTorch lightning + Weights &amp; Biases experiments. I made this template because I wanted to use a flexible YAML config file system, but larger templates like the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) seemed way too complex and a bit overkill for simpler projects.


## Setting up

```bash
# clone template
git clone https://github.com/jonathanswinnen/pl-template.git

cd pl-template

# create environment
python -m venv env
. env/bin/activate

# install requirements
pip install -r requirements.txt
```

## Working with the template

The main idea of this template is that you simply write your model, data module and other helper classes (data augmentations, PL Callbacks, utilities, etc.) first, and then easily run experiments with different parameters through the use of YAML config files.


The `experiment.PlExperiment` class represents such an experiment configered using a YAML file, and does all the setup (instantiating the datamodule, model and wandb logger, loading checkpoints, etc.).


This repository contains some very basic example code to illustrate the idea. (See `datamodule.py`, `model.py`, for a simple dummy datamodule and model, `configs/defaults.yaml` for some basic configuration and `test.ipynb` to see how to run experiments)


This is just a first very basic version of this template, and it is still evolving. 
Detailed documentation might be added in the future.
