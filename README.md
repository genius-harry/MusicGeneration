This project is based on another github project:
https://github.com/Skuldur/Classical-Piano-Composer

This project allows you to train a neural network to generate midi music files that make use of a single instrument based on a text input

## Requirements

* Python 3.x
* Installing the following packages using pip:
	* Music21
	* Keras
	* Tensorflow
	* h5py
	* openai

## Training

To train the network you run **lstm.py**.

E.g.

```
python lstm.py
```


**NOTE**: You can stop the process at any point in time and the weights from the latest completed epoch will be available


