# Generative Graph Injection Attacks on GNN Classifier
By Taylor Neller, Gaurav Shah, Dongha Yoon

## Overview
This project demonstrates a rudimentary graph injection attack on the MUTAG and PROTEINS datasets. The code will train a GAN to create semantically similar graphs to the training sets and then inject those graphs into the training data. The attack success is determined by the accuracy degradation of the victim classifier, which in our case is a GCN.

---

## Prerequisites
Before running the experiment, ensure you install the environment.yml environment (uses Python 3.12)

---

## Experiment 

To repeat our experiment, first compute the degree distribution for the dataset of your choice, using compute_distributions.py. Next, run either GCNModel.py or GCNModel2.py depending on the dataset you chose. This will generate the surrogate classifier model. For MUTAG, the associated files are TrainGAN.py, GCNModel.py, and InjectGCN.py, while for PROTEINS the associated files are TrainGAN2.py, GCNModel2.py, and InjectGCN2.py, as we did not have enough time to make the files flexible enough for both datasets. After generating the surroagate, TrainGAN.py or TrainGAN2.py depending on the dataset, and likewise run InjectGCN.py or InjectGCN2.py. For the MUTAG dataset injection (InjectGCN.py), there is also support for perfoming the evaluation with outlier filtering. After running the injection code, the accuracy of the victim classifier after injection will be displayed at the end of the last epochs. Feel free to adjust n_samples at the top of the injection code to change how many adversarial examples are injected.
To view the distribution statistics for any given dataset run CompareGraphs.py. Adjust the input according to the graphs you wish to compare. 
