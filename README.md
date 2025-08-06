# CIFAR10-WITH-VIT-and-CNN

Note: VIT8, VIT16 8 and 16 do not represent anything, just a random number at the end to signify versions
Data augmentation was used to create 20 different data sets of the same size as CIFAR-10(50k images), which are not uploaded here.
With augmentation, VIT8 got around 86% accuracy, and VIT16 got 81% accuracy. CNN got an accuracy of 90%
No other dataset was used to train them.
VIT16  has only one single global convolution basic block, then it's segmented into 8 x 8 image segments and passed through the attention blocks
