# R-GAE (Rethinking Graph Auto-Encoder Models for Attributed Graph Clustering)

## Abstract

Most recent graph clustering methods have resorted to Graph Auto-Encoders (GAEs) to perform joint clustering and embedding learning. However, two critical issues have been overlooked. First, the accumulative error, inflicted by learning with noisy clustering assignments, degrades the effectiveness and robustness of the clustering model. This problem is called Feature Randomness. Second, reconstructing the adjacency matrix sets the model to learn irrelevant similarities for the clustering task. This problem is called Feature Drift. Furthermore, the theoretical relation between the aforementioned problems has not yet been investigated. We study these issues from two aspects: (1) there is a trade-off between Feature Randomness and Feature Drift when clustering and reconstruction are performed at the same level, and (2) the problem of Feature Drift is more pronounced for GAE models, compared with vanilla auto-encoder models, due to the graph convolutional operation and the graph decoding design. Motivated by these findings, we reformulate the GAE-based clustering methodology. Our solution is two-fold. First, we propose a sampling operator that triggers a protection mechanism against the noisy clustering assignments. Second, we propose an operator that triggers a correction mechanism against Feature Drift by gradually transforming the reconstructed graph into a clustering-oriented one. As principal advantages, our solution grants a considerable improvement in clustering effectiveness and robustness and can be easily tailored to existing GAE models.

## Conceptual design

<p align="center">
<img align="center" src="https://github.com/nairouz/R-GAE/blob/master/image_2.png">
</p>

## Some results

### Quantitative 
<p align="center">
<img align="center" src="https://github.com/nairouz/R-GAE/blob/master/image_3.png" >
</p>
<p align="center">
<img align="center" src="https://github.com/nairouz/R-GAE/blob/master/image_4.png" >
</p>

### Qualitative 
<p align="center">
<img align="center" src="https://github.com/nairouz/R-GAE/blob/master/image_1.png">
</p>

## Usage

We provide four models GMM-VGAE, DGAE, R-GMM-VGAE, and R-DGAE. For each dataset and each model, we provide the pretraining weights. The data is also provided with the code.   Users can perform their own pretraining if they wish. GPU(s) are not required to train the models. For instance, to run the code of R-GMM-VGAE on Cora, you should clone this repo and use the following command: 
```
python3 ./R-GMM-VGAE/main_cora.py
```

## Built With

All the required libraries are provided in the ```requirement.txt``` file. The code is buildt with:

* Python 3.6
* Pytorch 1.7.0
* Scikit-learn
* Scipy

## Datasets

### Cora (https://relational.fit.cvut.cz/dataset/CORA): 
The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.


### Citeseer (https://linqs.soe.ucsc.edu/data): 
The citeseer dataset consists of 3312 scientific publications classified into one of six classes. The citation network consists of 4732 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 3703 unique words.


### Pubmed (https://linqs.soe.ucsc.edu/data): 
The Pubmed dataset consists of 19717 scientific publications from PubMed database pertaining to diabetes classified into one of three classes. The citation network consists of 44338 links. Each publication in the dataset is described by a TF/IDF weighted word vector from a dictionary which consists of 500 unique words.


### Brazilian air-traffic network (https://www.anac.gov.br/): 
Data collected from the National Civil Aviation Agency (ANAC) from January to December 2016. It has 131 nodes, 1,038 edges (diameter is 5). Airport activity is measured by the total number of landings plus takeoffs in the corresponding year.


### American air-traffic network (https://transtats.bts.gov/): 
Data collected from the Bureau of Transportation Statistics from January to October, 2016. It has 1,190 nodes, 13,599 edges (diameter is 8). Airport activity is measured by the total number of people that passed (arrived plus departed) the airport in the corresponding period.


### European air-traffic network (https://ec.europa.eu/): 
Data collected from the Statistical Office of the European Union (Eurostat) from January to November 2016. It has 399 nodes, 5,995 edges (diameter is 5). Airport activity is measured by the total number of landings plus takeoffs in the corresponding period.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

* **Nairouz Mrabah** - *Grad Student (Université du Québec à Montréal)* 
* **Mohamed Bouguessa** - *Professor (Université du Québec à Montréal)*
* **Mohamed Fawzi Touati** - *Grad Student (Université du Québec à Montréal)* 
* **Riadh Ksantini** - *Professor (University of Bahrain)*

 
## Citation
  
  ```
@ARTICLE {nmrabah,
author = {N. Mrabah and M. Bouguessa and M. Touati and R. Ksantini},
journal = {IEEE Transactions on Knowledge and Data Engineering},
title = {Rethinking Graph Auto-Encoder Models for Attributed Graph Clustering},
}
  ```
