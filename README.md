# Deep Conversion Attribution for Online Advertising
A `tensorflow` implementation of all the compared models for the KDD submission "Learning Conversion Attribution with Dual-attention Mechanism for Online Advertising".
If you have any questions, please feel free to contact [Kan Ren](http://apex.sjtu.edu.cn/members/kren) (kren@apex.sjtu.edu.cn) and [Yuchen Fang](http://apex.sjtu.edu.cn/members/arthur_fyc) (arthur_fyc@apex.sjtu.edu.cn).

### Data Preparation
We have upload a tiny data sample for training and evaluation.
The full dataset for this project can be download from this [link](http://apex.sjtu.edu.cn/datasets/13).
After download please replace the sample data in `data/` folder with the full data files.

### Installation and Running
[TensorFlow](https://www.tensorflow.org/)(>=1.2) and dependant packages (e.g., `numpy` and `sklearn`) should be pre-installed before running the code.

After package installation, you can simple run the code with the demo tiny dataset.
```
python LR.py [learning rate]                    # for LR
python AH.py                                    # for AdditiveHazard
python AMTA.py [learning rate] [batchsize]      # for AMTA
python ARNN.py [learning rate] [batchsize] [mu] # for ARNN
python DARNN.py [learning rate] [batchsize] [mu]# for DARNN
```

We have set default hyperparameter in the model implementation. So the parameter arguments are optional for running the code.
