## Abstract
Entity Matching (EM) defines the task of learning to group objects by transferring semantic concepts from example groups (=entities) to unseen data. Despite the general availability of image data in the context of many EM-problems, most currently available EM-algorithms solely rely on (textual) meta data.<br>
We introduce the first publicly available large-scale dataset for "visual entity matching", based on a production level use case in the retail domain. Using scanned advertisement leaflets, collected over several years from different European retailers, we provide a total of ~786k manually annotated, high resolution product images containing ~18k different individual retail products which are grouped into ~3k entities. The annotation of these product entities is based on a price comparison task, where each entity forms an equivalent class of comparable products.<br>
Following on a first baseline evaluation, we show that the proposed ``visual entity matching'' constitutes a novel learning problem which can not sufficiently be solved using standard image based classification and retrieval algorithms. Instead, novel approaches which allow to transfer example based visual equivalent classes to new data are needed to solve the proposed problem. The aim of this paper is to provide a benchmark for such algorithms.<br>
<br>
In the context of retail products, the term "visual entity matching" refers to the task of linking individual product images from diverse sources to a semantic product grouping. All images in the below figure show different products from the same entity which is defined by the fact that single images are used as ``placeholders`` by retailers to promote all products of the entity.<br>
<p align="center">
    <img src="/images/visual_abstract.svg">
</p>


## Dataset Description
This dataset is a **large-scale** dataset for "**visual entity matching**". Entity Matching (EM) indicates the identification of semantic groupings. As it occurs like the identification of consumer products in images in retail applications. A product is a retail item on sale that is uniquely identified by the internationally standardized Global Trade Item Number (GTIN). Entities are then defined as semantic groupings of products.

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
