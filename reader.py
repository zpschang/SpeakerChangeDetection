from classifier import SpeakerClassifier
import numpy as np
from parameter import parameter
from preprocess import get_feature
import os
import re
import random
flag_true = 0
flag_false = 0

# def get_feature(name):
#     return [[0] * parameter.feature_length for _ in range(100)]

class Reader():
    def __init__(self, is_train=True, is_cut=False):
        print 'load dataset'
        self.iter_list = []
        self.loss_list = []
        self.acc_list = []
        self.is_train = is_train
        self.is_cut = is_cut
        if self.is_train:
            self.audio_dir = 'dataset/wav/'
            self.label_dir = 'dataset/label/'
        elif not self.is_cut:
            self.audio_dir = 'dataset/wav_test/'
            self.label_dir = 'dataset/label_test/'
        else:
            self.audio_dir = 'dataset/wav_infer/'
            self.label_dir = None
        self.id = self.get_id()
        random.shuffle(self.id)
        self.speaker = []
        self.speaker2id = {}
        self.timeline = {}
        self.feature = {}

        self.epoch = 0
        self.iter = 0
        self.total_loss = self.total_acc = self.num = 0

        num = 0
        for id in self.id:
            if num % 20 == 0:
                print '%d / %d' % (num, len(self.id))
            num += 1
            self.timeline[id] = []
            feature = get_feature(self.audio_dir + id + '.wav')
            self.feature[id] = feature
            if self.is_cut:
                continue
            stream = open(self.label_dir + id + '.txt', 'rb')
            for line in stream.readlines():
                result = re.findall('[^ \t\n\r]+', line)
                if len(result) != 3:
                    continue
                start, end, speaker = result
                if speaker not in self.speaker2id:
                    self.speaker2id[speaker] = len(self.speaker)
                    self.speaker.append(speaker)
                self.timeline[id].append((eval(start), eval(end), self.speaker2id[speaker]))
        self.batch = []
        parameter.speaker_num = len(self.speaker)
        num = 0
        for id in self.id:
            print '%d / %d' % (num, len(self.id))
            num += 1
            timeline = None if self.is_cut else self.timeline[id]
            feature_list = self.feature[id]
            for time in range(len(feature_list) - parameter.block_size):
                # print time
                block_feature = []
                for t in range(time, time + parameter.block_size):
                    # print self.feature[id][t]
                    block_feature.append(self.feature[id][t])
                block_feature = np.array(block_feature)
                if self.is_cut:
                    self.batch.append((block_feature, None))
                    continue
                # feature = self.feature[id][time]
                block_time = parameter.block_ms / 1000
                realtime = time * block_time
                realtime_start = realtime
                realtime_end = realtime + parameter.block_size * block_time
                flag = False
                for start, end, speaker in timeline:
                    if end <= realtime_start:
                        continue
                    elif start <= realtime_start and realtime_end-block_time <= end:
                        flag = True
                        break
                    elif start <= realtime_start and end < realtime_end-block_time:
                        flag = False
                        break
                # print flag, '(%f, %f) (%f, %f)' % (start, end, realtime_start, realtime_end)
                if not flag:
                    continue
                # if 'SIL' in self.speaker2id and speaker == self.speaker2id['SIL']:
                #     continue
                self.batch.append((block_feature, speaker))
        if self.is_train:
            random.shuffle(self.batch)
        parameter.correct = [0 for _ in range(len(self.speaker))]
        parameter.total = [0 for _ in range(len(self.speaker))]
        parameter.predict = [0 for _ in range(len(self.speaker))]
        print len(self.speaker)
        print 'load finished'

    def get_id(self):
        audio = set(os.listdir(self.audio_dir))
        label = set(os.listdir(self.label_dir)) if not self.is_cut else None
        id_list = []
        for audio_file in audio:
            id = audio_file[:-4]
            if id == 'M034PNI_F067ANV_006':
                continue
            if id == '.DS_S':
                continue
            label_file = id + '.txt'
            if self.is_cut or label_file in label:
                id_list.append(id)
        return id_list

    def get_batch(self):
        feature_list = []
        speaker_list = []
        for _ in range(parameter.batch_size):
            feature, speaker = self.batch[self.iter]
            feature_list.append(feature)
            speaker_list.append(speaker)

            self.iter += 1
            if self.is_train and self.iter % 1000 == 0:
                for index in range(len(self.speaker)):
                    print '%s acc: %d / %d, %d' % (self.speaker[index], parameter.correct[index], parameter.total[index], parameter.predict[index])
                print 'epoch: %d, iteration: %d' % (self.epoch, self.iter)
                print 'avg loss: %f, avg acc: %f' % (self.total_loss / self.num, self.total_acc / self.num)
                self.iter_list.append(self.iter + len(self.batch) * self.epoch)
                self.loss_list.append(self.total_loss / self.num)
                self.acc_list.append(self.total_acc / self.num)
                self.total_loss = self.total_acc = self.num = 0
            if self.iter == len(self.batch):
                self.iter = 0
                self.epoch += 1
        return feature_list, speaker_list


    def record(self, loss, acc):
        self.total_loss += loss
        self.total_acc += acc
        self.num += 1
