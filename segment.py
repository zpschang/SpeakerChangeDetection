from classifier import SpeakerClassifier
from reader import Reader
from parameter import parameter
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    exit(0)
cmd = sys.argv[1]
is_cut = (cmd == "infer")

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

model = SpeakerClassifier()
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=1.0)
if True:
    loader = tf.train.import_meta_graph('model/model.ckpt.meta')
    loader.restore(sess, tf.train.latest_checkpoint('model/'))
    print 'load finished'
#except:
#    print 'load failed'

reader_test = Reader(is_train=False, is_cut=is_cut)

parameter.batch_size = 1

def distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

prev_result = None
prev_speaker = None
file_output = open('result.txt', 'w')

change_list = []
identical_list = []
time_list = []
distance_list = []
total = 0
correct = 0
if not is_cut:
    while reader_test.epoch < 1:
        batch = reader_test.get_batch()
        result = model.evaluate(sess, batch)
        result = result[0]
        if not prev_result is None:
            d = distance(result, prev_result)
            time_list.append(reader_test.iter * parameter.block_ms)
            distance_list.append(d)
            file_output.write('distance: ' + str(d))
            speaker = batch[1][0]
            file_output.write(' ' + str(prev_speaker) + '->' + str(speaker) + '\n')
            total += 1
            if speaker != prev_speaker:
                change_list.append(d)
                correct += 1 if d >= 0.5 else 0
            else:
                identical_list.append(d)
                correct += 1 if d < 0.5 else 0
        argmax = np.argmax(result)
        file_output.write('time: %f ' % (reader_test.iter * parameter.block_ms / 1000))
        # file_output.write('max: %s, p = %f\n' % (reader_test.speaker[argmax], result[argmax]))
        prev_result = result
        prev_speaker = batch[1][0]

    print 'change:', 0 if len(change_list) == 0 else sum(change_list) / len(change_list)
    print 'identical:', sum(identical_list) / len(identical_list)
    print 'total segmentation accuracy:', 1.0 * correct / total

    plt.hist(change_list, bins=14, range=(0,1.4), normed=1, color=(1,0,0,0.5))
    plt.hist(identical_list, bins=14, range=(0, 1.4), normed=1, color=(0,1,0,0.5))
    plt.savefig('identical.png')
    plt.clf()

    plt.xlabel('time')
    plt.ylabel('distance')
    plt.plot(time_list, distance_list, label='distribution distance')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    fig.savefig('distance.png', dpi=100)
    print 'test end'
else:
    while reader_test.epoch < 1:
        batch = reader_test.get_batch()
        result = model.evaluate(sess, batch)
        result = result[0]
        if not prev_result is None:
            d = distance(result, prev_result)
            if d > 0.5:
                file_output.write('time: %f, speaker change\n' % (reader_test.iter * parameter.block_ms / 1000))
        prev_result = result
    print 'infer end'