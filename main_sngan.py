import tensorflow as tf
import os
from SNGAN import *

flags = tf.app.flags
flags.DEFINE_bool("is_training", True, "training flag")
FLAGS = flags.FLAGS

def check_dir():
    if not os.path.exists('./sample'):
        os.mkdir('./sample')
    if not os.path.exists('./checkpoint'):
        os.mkdir('./checkpoint')
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

if __name__=='__main__':
    check_dir()
    with tf.Session() as sess:
        GAN = SNGAN(input_height=32, input_width=32, input_channels=3, output_height=32, output_width=32, gf_dim=32,df_dim=32,
                    input_fname_pattern = '*.jpg', is_grayscale=False, sess = sess)
        GAN.build_model()
        if FLAGS.is_training:
            print("training is starting!!!")
            GAN.train()
            print("training is ended!!!")
