# basket-hog
 1. get positive sample and negative sample from videos
 2. using hog+svm to train my own classifier/model
 3. using classifier to detect and get the hard negative sample
 4. retrain and get new classifier
 5. using classifier to detect basket in videos

## File usage
 - negative folder: negative sample picture, generated by `get_negative_set.py` 
 - positive folder: positive sample picture, generated by `get_positive_set.py` 
 - neg_hard folder: negative sample picture by retrain once 
 - pos_hard folder: positive sample picture by retrain once
 - neg_hard2 folder: negative sample picture by retrain twice
 - pos_hard2 folder: positive sample picture by retrain twice
 - test folder: predict samples 
 - test_performance folder: test samples for classifier
 - data_*.xml: model by different number of samples
 - util.py: draw rectangle, get vector from xml, used by other *.py 
 - config.txt: configure file, actually it's not used 
 - data.model: generated by libsvm_train.py 
 - libsvm_*.py: use libsvm 
 - detect_*.py: detect target by opencv svm
 - get_hard.py: get hard example by detect 
 - nms.py: to get one rectangle from several rectangle 

