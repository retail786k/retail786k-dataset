# Retail-786k Dataset

## Abstract
Lorem ipsum dolor sit amet, consectetur adipisici elit, ...<br>
INCLUDE visual abstract<br>
![Visual Abstract](/images/YYY.png)

## Dataset Description
Lorem ipsum dolor sit amet, consectetur adipisici elit, ...

## Download Dataset
todo: Link zenodo<br>
The dataset can be downloaded via the link: [Retail-786k Dataset](https://zenodo.org/record/XXX.XXX)

Download the dataset via a Jupyter-Notebook:<br>
todo: Links anpassen<br>
Jupyter-Notebook
```
!pip install wget

# version 256
!python -m wget https://zenodo.org/record/XXX/files/X.zip?download=1?

# version 512
!python -m wget https://zenodo.org/record/XXX/files/X.zip?download=1?
```
## Code for baseline experiments
It is assumed that the data is downloaded in the same working directory.

### Image Classification

### Image Retrieval

We use the following algorithm for the image retrieval task: [ROADMAP](https://github.com/elias-ramzi/ROADMAP)<br>
Please follow the instructions on the website to use ROADMAP.<br>

```
conda activate roadmap_env

cd ROADMAP

# running ROADMAP algorithm
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES='0,1,2,3' python roadmap/single_experiment_runner.py 'experience.experiment_name=retail-786k_256_ROADMAP_${dataset.sampler.kwargs.batch_size}_sota' experience.seed=333 experience.max_iter=100 'experience.log_dir=.' optimizer=retail-786k model=resnet transform=retail-786k_256 dataset=retail-786k_256 dataset.sampler.kwargs.batch_size=128 loss=roadmap
```

## License
This dataset is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/)
