import tensorflow as tf

def configure():
    
    flags = tf.app.flags
    
    flags.DEFINE_string('dataName', './16_08_III_HE.mrxs', 'data location (file .mrxs)')
    flags.DEFINE_integer('priority', 1, 'Priority information of TMA location: -1: unknown, 1: normal, 2: tumor')
    
    flags.DEFINE_integer('level', 2, 'level factor for data image')
    
    
    flags.DEFINE_string('modelCNNLocation', './Pre-trained/CNN/Lenet_weights-improvement-347-0.99.h5', 'pre-trained model CNN dir')
    
    flags.DEFINE_string('modelCapsNetLocation', './Pre-trained/CapsNet/c1s_9_c1n_256_c2s_6_c2n_64_c2d_0.7_c1vl_16_c1s_5_c1nf_16_c2vl_32_lr_0.0001_rs_1--HistoPath--1563806826.3830802', 'pre-trained model CapsNet dir')
    flags.DEFINE_string('modelName', 'HistoPath', 'model name of CapsNet')
    
    flags.DEFINE_integer('nb_label', 3, '# number of classes')
    
    flags.DEFINE_integer('maxAreaTMASpot', 2000, 'threshold for the size of TMA spot (max)')
    flags.DEFINE_integer('minAreaTMASpot', 100, 'threshold for the size of TMA spot (min)')
    
    return flags.FLAGS

