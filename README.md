# Implementation: Deep learning for classification of colorectal tissue TMA images 


## Requirements
- Python 3
- openslide
- NumPy 1.13.1
- Tensorflow 1.3.0
- Sklearn: 0.18.1
- Matplotlib

   
## Train

    $> python train.py -h
    $> python train.py dataset/

During the training, the checkpoint is saved by default into the outputs/checkpoints/ folder. The exact path and name of the checkpoint is print during the training.

## Test

In order to measure the accuracy and the loss on the Test dataset you need to used the test.py script as follow:

    $> python test.py outputs/checkpoints/ckpt_name dataset/ 


## Example of result

<img src="images/tensorboard.png"></img>




