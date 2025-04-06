# perceptrons-to-transformers

## 0. Create Environemnt

```SHELL
conda env create -f environment.yml
conda activate dnn
```

## 1. Perceptrons
To run a single layer perceptron on AND
```SHELL
python main.py -s 0
```
To run a single layer perceptron on XOR
```SHELL
python main.py -s 1
```
To run a multi layer perceptron on XOR
```SHELL
python main.py -s 2
```

Feel free to play around with the learning rate! The default is 0.1 and only values 0 to 1 are allowed.
```SHELL
python main.py -s 0 -r 0.2
```
