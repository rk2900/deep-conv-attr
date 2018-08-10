# Deep Conversion Attribution for Online Advertising
A `tensorflow` implementation of all the compared models for the CIKM 2018 paper: Learning Multi-touch Conversion Attribution with Dual-attention Mechanisms for Online Advertising.

### Data Preparation
We have uploaded a tiny data sample for training and evaluation.
The full dataset for this project will be published soon.

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
