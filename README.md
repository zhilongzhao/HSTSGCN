***First, the structure of the code:

util.py: The reading of data and other functions except of Model

utils.py: all base blocks and functions of neural network models in our paper

model.py: construct model based on utils.py

engine.py: the program of training model in our paper

train_h.py: HSTSGCN

***How to run these files?

In jupyter ,you should write:

run train_h.py --model (you can select, such as HSTSGCN) --force True

If you want to change the dataset from XiAn_city to JiNan_city, I suggest you can directly revise the code in your IDE in train_h.py

See the paper for an analysis of hyperparameters.

if you want get the dataset from XiAn_city to JiNan_city.

the HSTSGCN result can be seen as follow;
