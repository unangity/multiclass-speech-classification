# Multiclass Speech Classification

This repository contains code and associated files for classifying speech recordings.

## Overview
The task is to train a classifier to recognize which of several English words is pronounced in an audio recording.

## Data
The dataset used for this task can be found at https://surfdrive.surf.nl/files/index.php/s/A91xgk7B5kXNvfJ which contains the following files:
* wav.tgz: a compressed directory with all the recordings (training and test data) in the form of wav files.
* feat.npy: an array with Mel-frequency Cepstral Coefficients (MFCC) extracted from each wav file. The features at index i in this array were extracted from
the wav file at index i of the array in the file path.npy.
* path.npy: an array with the order of wav files in the feat.npy array.
* train.csv: this file contains two columns: path with the filename of the recording and word with word which was pronounced in the recording. This is the training portion of the data.
* test.csv: This is the testing portion of the data, and it has the same format as the file train.csv except that the column word is absent.

In this task, the classifier examines the MFCC features extracted and performs a multiclass classification; labeling each feature as one of 35 English words