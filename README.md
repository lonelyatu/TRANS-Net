# TRANS-Net
A novel reconstruciton framework for MR imaging.


TRANS-Net is an unrolled reconstruction method integrated with transformer module, which additionally employs the regularization term on the residual image and get promising improvement.


# Training
1. run "python TFRecordOp.py" and the files in the "TrainFile" will be saved as tfrecord files

2. run "python Train_TRANSNet.py" and the model will be saved every epoch

# Testing
1. run "python Test_TRANSNet.py" to test the files in the "TestFile"

# Environment

cuda 10.0
python 3.6.13
TensorFlow 1.15.4
Numpy 1.16.0
Scipy 1.2.1


# Description
Due to the limitation, we only provide 100 images to train and test. You can use your only datasets to train and test.

Also, we only provide the Gaussian 2D undersampling pattern with acceleration factor 10, you can use your own masks.


