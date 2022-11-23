# fake-reviews-detection

The objective of this project is to use machine learning techniques to classify user reviews obtained from 'Deceptive Opinion Spam Corpus' dataset as deceptive or thruthful and evaluate the performance of different approaches.

Repository Structure:
    
    source
    ├── bag-of-words
    │   └── bag_of_words_classifier.ipynb -> Colab notebook implementing PCA-SVM model and Random Forest model
    ├── bert
    │   └── bert_classifier.ipynb         -> Colab notebook implementing BERT model
    ├── common
    │   ├── data_preprocessing.py         -> process raw data and extract necessary features
    │   ├── linguistic_feature.py         -> extract linguistic features
    │   ├── ngram.py                      -> find ngrams
    │   └── sentence_contractions.py      -> dictionary for sentence contractions
    └── dataset-analysis
        └── data_analysis.ipynb           -> analyze data by plotting various histograms

    evaluations: contains screenshots of results and plots used for model and dataset analysis
    
    data: contains data obtained from 'Deceptive Opinion Spam Corpus'
    
How to execute:
This repository is designed to be executed on Google Colab platform.
Clone the repository and upload it to Google Drive under 'Colab Notebooks' directorty.
Navigate to the required *.ipynb Colab notebook to be executed and launch it in Colab.

Execute outside Google Colab on a local machine:
The source code expects entire repository to be present with a root path '/content/drive/My Drive/Colab Notebooks/'
All the paths for loading/saving data uses this root path as the prefix.
Remove the root path prefixes from all the references and then the scripts can be executed in Jupyter Notebook.
