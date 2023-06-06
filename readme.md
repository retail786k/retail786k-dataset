## Abstract
Lorem ipsum dolor sit amet, consectetur adipisici elit, ...<br>
<br>
In the context of retail products, the term ``Visual Entity Matching'' refers to the task of linking individual product images from diverse sources to a semantic product groupings. The following image illustrates such grouping.<br>
<p align="center">
    <img src="/images/visual_abstract.svg">
</p>


## Dataset Description
This dataset is a **large-scale** dataset for ``**visual entity matching**''. Entity Matching (EM) indicates the identification of semantic groupings. As it occurs like the identification of consumer products in images in retail applications. A product is a retail item on sale that is uniquely identified by the internationally standardized Global Trade Item Number (GTIN). Entities are then defined as semantic groupings of products.

The dataset consists of **786,179** images labeled with 3,298 different entities. The images are split into sub-sets of 748,715 training images and 37,464 test images. The images of the dataset are cropped from scanned advertisement leaflets, collected over several years from different European retailers.

Two versions of the dataset are provided: once with the longer edge fixed to 512 and another one fixed to 256.

## Download Dataset
The dataset can be downloaded via the link: [Retail-786k Dataset](https://zenodo.org/record/7970567)

To download the dataset via a Jupyter-Notebook:
```
!pip install wget

# version 256
!python -m wget https://zenodo.org/record/7970567/files/retail-786k_256.zip?download=1

# version 512
!python -m wget https://zenodo.org/record/7970567/files/retail-786k_512.zip?download=1
```
## Code for baseline experiments
For both experiments, it is assumed that the dataset is downloaded in the same working directory.

### Image Classification
Run [static_classification.ipynb](code/classification/static_classification.ipynb) to get the results for the ``Static'' EM as Classification Problem.

### Image Retrieval

Copy the <em>retail-786k_256_info_all_*.txt</em> files into the dataset folder <em>retail-786k_256</em><br>
(the files must be at the same folder level as the train and test folder)

```
cp retail-786k_256_info_all_train.txt retail-786k_256
cp retail-786k_256_info_all_test.txt retail-786k_256
```

We use the following algorithm for the image retrieval task: [ROADMAP](https://github.com/elias-ramzi/ROADMAP)<br>
First, clone the repository.<br>
Copy the files of the folder <em>roadmap</em> of this repository into the corresponding folder of the downloaded <em>ROADMAP</em> repository.
```
cp -r roadmap ROADMAP
```
Add the path to dataset in the file <em>roadmap/config/dataset/retail786k_256.yaml</em><br>
Follow the instructions on the website to use ROADMAP.<br>
Afterwards, run the following commands:
```
cd ROADMAP

# running ROADMAP algorithm
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES='0,1,2,3' python roadmap/single_experiment_runner.py 'experience.experiment_name=retail786k_256_ROADMAP_${dataset.sampler.kwargs.batch_size}_sota' experience.seed=333 experience.max_iter=100 'experience.log_dir=.' optimizer=retail786k_256 model=resnet transform=retail786k_256 dataset=retail786k_256 dataset.sampler.kwargs.batch_size=128 loss=roadmap
```

## License
This dataset is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/)
