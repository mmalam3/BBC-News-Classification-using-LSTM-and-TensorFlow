# BBC News Classification using LSTM and TensorFlow

This repository contains the implementation of an NLP-based **Text Classifier** that classifies a set of BBC News into multiple categories. It is developed using `TensorFlow`, `LSTM`, `Keras`, `Scikit-Learn`, and `Python`. The goal of this project is to perform Natural Language Processing (NLP) over a collection of texts compiled from BBC News, teach the classifer about the text features, and determine the appropriate class given a news text from a test dataset. 

The classifier designed in this project is a deep neural network with one `Embedding` layer, two `LSTM` layers, and one output `Dense` layer. The optimzer and the loss function chosen for the model are `Adam` and `sparse_categorical_crossentropy` respectively.

## Dataset

The [BBC News Classification](https://www.kaggle.com/competitions/learn-ai-bbc/data) dataset is used in this project for training and testing the models. The dataset comprises of 2225 articles, each labeled under one of 5 categories: `business`, `entertainment`, `politics`, `sport` or `tech`. It is parted into two sets: 1) train set with 1490 records, and 2) test set with 735 records.

## Usage

To run this project, open the notebook [bbc_news_classification.ipynb](https://github.com/kayanmorshed/BBC-News-Classification-using-LSTM/blob/main/bbc_news_classification.ipynb) in Google Colab and start executing each cell as instructed. The notebook contains detailed instructions about how to download the dataset correctly from Kaggle, and the only thing you need is to download the `kaggle.json` file from your account.  


## Evaluation

The following figure shows how training and validation accuracy change over epochs during the training process. It also represents the changes in training and validation loss over the training epochs.

![Accuracy and loss changes over training epochs](https://github.com/kayanmorshed/BBC-News-Classification-using-LSTM-and-TensorFlow/blob/main/evaluation/training_evaulations.png)


## Conclusion

This NLP-based News Classifier can be used to categorize any sets of news texts and determine the appropriate category of an unknown news.   
