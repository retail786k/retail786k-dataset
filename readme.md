# Retail-786k Dataset

## Abstract
Lorem ipsum dolor sit amet, consectetur adipisici elit, ...<br>
<br>
In the context of retail products, the term ``Visual Entity Matching'' refers to the task of linking individual product images from diverse sources to a semantic product groupings. The following image illustrates such grouping.<br>
<p align="center">
    <img src="/images/visual_abstract.svg">
</p>


## Dataset Description
Lorem ipsum dolor sit amet, consectetur adipisici elit, ...

## Download Dataset
todo: Link zenodo<br>
The dataset can be downloaded via the link: [Retail-786k Dataset](https://zenodo.org/record/XXX.XXX)

To download the dataset via a Jupyter-Notebook:<br>
todo: Links anpassen<br>
```
!pip install wget

# version 256
!python -m wget https://zenodo.org/record/XXX/files/X.zip?download=1?

# version 512
!python -m wget https://zenodo.org/record/XXX/files/X.zip?download=1?
```
## Code for baseline experiments
It is assumed that the dataset is downloaded in the same working directory.

### Image Classification
Run [static_classification.ipynb](code/classification/static_classification.ipynb) to get the results for the ``Static'' EM as Classification Problem.

### Image Retrieval

We use the following algorithm for the image retrieval task: [ROADMAP](https://github.com/elias-ramzi/ROADMAP)<br>
Please follow the instructions on the website to use ROADMAP.<br>

before run command:<br>
copy *.txt files into the dataset folder, same folder level as train and test<br>
copy the files in the folder roadmap_retail-786k in the corresponding folder of the downloaded ROADMAP repository<br>

```
conda activate roadmap_env

cd ROADMAP

# running ROADMAP algorithm
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES='0,1,2,3' python roadmap/single_experiment_runner.py 'experience.experiment_name=retail-786k_256_ROADMAP_${dataset.sampler.kwargs.batch_size}_sota' experience.seed=333 experience.max_iter=100 'experience.log_dir=.' optimizer=retail-786k model=resnet transform=retail-786k_256 dataset=retail-786k_256 dataset.sampler.kwargs.batch_size=128 loss=roadmap
```

## License
This dataset is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/)
