# CIFAR-10 Image Classification

## 1. Project Overview
This project aims to build an image classification model using the CIFAR-10 dataset.
A baseline CNN model will be implemented and improved through data augmentation
and model architecture changes.

## 2. Motivation
- To understand CNN-based image classification
- To observe performance changes by preprocessing and model modification
- To gain hands-on experience with an end-to-end deep learning pipeline

## 3. Dataset
- CIFAR-10
- 60,000 images (32x32 RGB)
- 10 classes

## 4. Project Plan
1. Load and explore the dataset
2. Build a baseline CNN model
3. Evaluate performance and identify issues
4. Improve the model (augmentation / ResNet)
5. Analyze results and document findings

## 5. Data Exploration
The CIFAR-10 dataset was loaded and sample images were visualized to understand
image resolution and class distribution before building the baseline model.

## 6. Baseline Model
A simple CNN with two convolutional layers was implemented as a baseline model.
This model serves as a reference point for evaluating future improvements.

## 7. Baseline Evaluation (Step 4)
Baseline CNN was trained for 5 epochs.

- Final Train Accuracy: 71.58%
- Final Test Accuracy:  68.09%

### Observations
- Test accuracy increased across the first few epochs (56.60% → 68.26%), and then slightly dropped at the last epoch (68.09%).
- The final train–test gap was small (about 3.49%p), suggesting limited overfitting and a reasonably stable baseline.

### Next Improvement Hypotheses
1) Data augmentation may further improve generalization by increasing effective training diversity.
2) A stronger architecture (e.g., ResNet) may capture more robust features on CIFAR-10.
