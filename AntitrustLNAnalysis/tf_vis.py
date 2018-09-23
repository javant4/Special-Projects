import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import gensim
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


fname = 'fr_model_w20'

model = gensim.models.keyedvectors.KeyedVectors.load(fname)
max_size = len(model.wv.vocab) - 1

w2v = np.zeros((max_size, model.layer1_size))

with open("metadata.tsv", 'w+') as file_metadata:
    for i, word in enumerate(model.wv.index2word[:max_size]):
        w2v[i] = model.wv[word]
        file_metadata.write(word + "\n")


sess = tf.InteractiveSession()
with tf.device('/cpu:0'):
    embedding = tf.Variable(w2v, trainable=False, name='embedding')


tf.global_variables_initializer().run()

path = 'tensorboard_fr_window'

saver = tf.train.Saver()
writer = tf.summary.FileWriter(path, sess.graph)


config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = 'embedding'
embed.metadata_path = 'metadata.tsv'

projector.visualize_embeddings(writer, config)
saver.save(sess, path+'/model.ckpt', global_step=max_size)

