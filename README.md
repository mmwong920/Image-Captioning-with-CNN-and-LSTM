# PA 3: Image Captioning

## Contributors
* Edwin Tse <ptse@ucsd.edu>
* Moses Wong <mmwong@ucsd.edu>
* Janet Lin <xil130@ucsd.edu>

## Task
In this assignment, our main goal is to perform Image Captioning task using CNNs and LSTMs and an encoder-decoder architechture. That is, CNN (an encoder) will take the image as input and encode it into a vector of feature values. Then will be passed through a linear layer for providing the input to an LSTM (a decoder). 

There are 2 tasks in total performed in this assignment, they are CNN + LSTM model and Resnet + LSTM model, where each model contains 3 individual sub tasks corresponding to different LSTM parameters. All of the results can be found in the task report. 

### An example of our prediction task looks like the following: 

![alt text](https://i.postimg.cc/zBWWRLrP/COCO-train2014-000000417570.jpg)

Prediction: A woman is riding a horse

## How to run

In terminal, simply run the following code to start the modeling process
```
 python3 main.py
```



## Usage
* Make sure the following config files are properly stored in the same directory: 
-- task-1-600emsize-config.json
-- task-1-default-config.json
-- task-1-1024hidden-config.json
* Simply run `python3 main.py` to start the experiment
* The logs, stats, plots and saved models, accuracies for each model will be generated after `main.py` is finished running.

## Files
- `main.py`: Main driver class
- `experiment.py`: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- `dataset_factory.py`: Factory to build datasets based on config
- `model_factory.py`: Factory to build models based on config
- `file_utils.py`: utility functions for handling files
- `caption_utils.py`: utility functions to generate bleu scores
- `vocab.py`: A simple Vocabulary wrapper
- `coco_dataset.py`: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- `get_datasets.ipynb`: A helper notebook to set up the dataset in your workspace
