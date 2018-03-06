from __future__ import division
from pydub import AudioSegment

import os, sys

# turn mp3 to wav
# AudioSegment.converter = '../ffmpeg-git-20171215-64bit-static/ffmpeg'

dir = "dataset/audio/"
index = 0
for i in os.listdir(dir):
    index += 1
    print index / len(os.listdir(dir))
    sound = AudioSegment.from_file(dir + i, format="mp3")
    file_handle = sound.export("dataset/wav/" + i[:-4] + '.wav', format="wav")
