from classifier import SpeakerClassifier
from reader import Reader
import time
from parameter import parameter
import os
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

reader = Reader()
model = SpeakerClassifier()
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=1.0)

try:
    loader = tf.train.import_meta_graph('model/model.ckpt.meta')
    loader.restore(sess, tf.train.latest_checkpoint('model/'))
    print 'load finished'
except:
    sess.run(tf.global_variables_initializer())
    print 'load failed'
try:
    while reader.epoch < 10000:
        batch = reader.get_batch()
        loss, acc = model.train(sess, batch)
        reader.record(loss, acc)
except KeyboardInterrupt:
    pass
saver.save(sess, 'model/model.ckpt')

plt.xlabel('iterations')
plt.ylabel('loss')
plt.plot(reader.iter_list, reader.loss_list, label='train loss')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(12, 12)
fig.savefig('loss.png', dpi=100)
plt.clf()
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.plot(reader.iter_list, reader.acc_list, label='train accuracy')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(12, 12)
fig.savefig('acc.png', dpi=100)