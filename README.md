# Deep Conversion Attribution for Online Advertising
A `tensorflow` implementation of all the compared models for the CIKM 2018 paper: Learning Multi-touch Conversion Attribution with Dual-attention Mechanisms for Online Advertising.

Paper Link: [https://arxiv.org/abs/1808.03737](https://arxiv.org/abs/1808.03737).

### Data Preparation
We have uploaded a tiny data sample for training and evaluation in this repository.

The full dataset for this project has been published [here](http://apex.sjtu.edu.cn/datasets/13).

After downloading please replace the sample data in data/ folder with the full data files.

### Data description
Our raw data is Criteo Attribution Modeling for Bidding Dataset . You can download it and read its description on [this page](http://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/).

Below are the descriptions of our data preprocessing.

1. We group all the impressions by user_id+conversion_id ( regard as one sequence ), shuffle the whole dataset, and then divide it into trainset and testset ( ratio: train 0.8, test 0.2) with negative down sampling (ratio 0.7) at the meanwhile.

2. We create mapping from features from certain fields ([campaign, cat1, cat2, …, cat9]) to index.

3. We turn every line into such format: “time click campaign cat1 cat2 … cat9”

### Installation and Running
[TensorFlow](https://www.tensorflow.org/)(>=1.2) and dependant packages (e.g., `numpy` and `sklearn`) should be pre-installed before running the code.

After package installation, you can simple run the code with the demo tiny dataset.
```
python LR.py [learning rate]                    # for LR
python SP.py                                    # for Simple Probablistic
python AH.py                                    # for AdditiveHazard
python AMTA.py [learning rate] [batchsize]      # for AMTA
python ARNN.py [learning rate] [batchsize] [mu] # for ARNN
python DARNN.py [learning rate] [batchsize] [mu]# for DARNN
```

We have set default hyperparameters in the model implementation. So the parameter arguments are optional for running the code.

### Citation
```
@inproceedings{ren2018learning,
  title={Learning Multi-touch Conversion Attribution with Dual-attention Mechanisms for Online Advertising},
  author={Ren, Kan and Fang, Yuchen and Zhang, Weinan and Liu, Shuhao and Li, Jiajun and Zhang, Ya and Yu, Yong and Wang, Jun},
  booktitle={Proceedings of the 27th ACM International on Conference on Information and Knowledge Management},
  year={2018},
  organization={ACM}
}
```
