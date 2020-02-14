# Implementation: Deep learning for classification of colorectal tissue TMA images 


## Requirements
- Python 3
- NumPy 1.13.1
- Tensorflow 1.3.0
- docopt 0.6.2
- Sklearn: 0.18.1
- Matplotlib

## Install

    $> git clone https://github.com/thibo73800/capsnet_traffic_sign_classifier.git
    $> cd capsnet_traffic_sign_classifier.git
    $> wget https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip
    $> unzip traffic-signs-data.zip
    $> mkdir dataset
    $> mv *.p dataset/
    $> rm traffic-signs-data.zip
   
## Train

    $> python train.py -h
    $> python train.py dataset/

During the training, the checkpoint is saved by default into the outputs/checkpoints/ folder. The exact path and name of the checkpoint is print during the training.

## Test

In order to measure the accuracy and the loss on the Test dataset you need to used the test.py script as follow:

    $> python test.py outputs/checkpoints/ckpt_name dataset/ 

## Metrics / Tensorboard

<b>Accuracy: </b>
<ul>
    <li>Train: 99%</li>
    <li>Validation: 98%</li>
    <li>Test: 97%</li>
</ul>

Checkpoints and tensorboard files are stored inside the <b>outputs</b> folder.

<img src="images/tensorboard.png"></img>

Exemple of some prediction:

<img src="images/softmax.png"></img>



