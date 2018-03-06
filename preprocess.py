from __future__ import print_function
import contextlib
import sys, os
import wave
import audioop
import webrtcvad
import numpy as np
import pickle
import time
from parameter import parameter

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def encode(s):
    return ''.join([bin(ord(c)).replace('0b', '').zfill(8) for c in s])


def get_feature(wav_id):
    with contextlib.closing(wave.open(wav_id, 'rb')) as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        data = wf.readframes(wf.getnframes())
        converted = audioop.ratecv(data, 2, 1, sample_rate, 16000, None)

    vad = webrtcvad.Vad(0)
    frames = frame_generator(parameter.block_ms, converted[0], 16000)
    frames = list(frames)
    # print(len(frames))
    # sum = 0
    # last = False
    # for i in frames:
    #   if(not vad.is_speech(i.bytes, 16000)):
    #       sum += 1
    #       if(last == False):
    #           print(str(i.timestamp) + "---", end = '')
    #           last = True
    #       else:
    #           pass
    #   else:
    #       if(last == True):
    #           print(str(i.timestamp - i.duration))
    #           last = False
    # print(sum)
    #feature_list = []
    index = 0
    for frame in frames:
        feature = np.fromstring(frame.bytes, dtype=np.short).astype(np.float32) / 32768
        if(index == 0):
            feature_list = np.zeros((len(frames), feature.shape[0]))
        feature_list[index, :] = feature
        index += 1
        #frequency = dft(feature)
        #feature_list.append(frequency[:frequency.shape[0] / 4])
    #print(dft(feature_list)[:, :feature_list.shape[1] / 4].shape)
    return abs(np.fft.fft(feature_list)[:, :feature_list.shape[1] / 4])

def dft(time_domain):
    length = time_domain.shape[-1]
    i, j = np.meshgrid(np.arange(length), np.arange(length))
    w = np.power(np.exp(-2j * np.pi / length), i * j)
    frequency_domain = np.dot(time_domain, w)
    return abs(frequency_domain)

if __name__ == '__main__':
    start_time = time.time()
    f = get_feature('F001HJN_F002VAN_001')
    print(time.time() - start_time)
    # temp = np.array([[1, 2, 4, 3, 2], [2, 3,44, 5, 6]])
    # print(dft(temp))
    # print(abs(np.fft.fft(temp)))
    #print(dft(f[250]).shape[0])
#print(encode(frames[0].bytes))

# sum = 0
# last = False
# for i in frames:
# 	if(not vad.is_speech(i.bytes, 16000)):
# 		sum += 1
# 		if(last == False):
# 			print(str(i.timestamp) + "---", end = '')
# 			last = True
# 		else:
# 			pass
# 	else:
# 		if(last == True):
# 			print(str(i.timestamp - i.duration))
# 			last = False
# print(sum)


# with contextlib.closing(wave.open("./1.wav", 'wb')) as wb:
# 	wb.setnchannels(1)
# 	wb.setsampwidth(2)
# 	wb.setframerate(16000)
# 	wb.writeframes(converted[0])
# print(num_channels)
# print(sample_width)
# print(sample_rate)