# Deep Conversion Attribution for Online Advertising
A `tensorflow` implementation of all the compared models for the working paper.

### Data Preparation
We have upload a tiny data sample for training and evaluation.
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

We have set default hyperparameter in the model implementation. So the parameter arguments are optional for running the code.
