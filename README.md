# TRANS-Net
A novel reconstruciton framework for MR imaging.


TRANS-Net is an unrolled reconstruction method integrated with transformer module, which additionally employs the regularization term on the residual image and get promising improvement.


# Training
1. run "python TFRecordOp.py" and the files in the "TrainFile" will be saved as tfrecord files

2. run "python Train_TRANSNet.py" and the model will be saved every epoch

# Testing

due to the limitation, we only provide 100 images to 
