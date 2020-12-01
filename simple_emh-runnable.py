#!/usr/bin/python3
# you can prepare the training dataset like so:
# ffmpeg -i full.wav -f segment -segment_time 1 -c copy out%03d.wav
# and then manually dividing the speech from emh(s) into separate folders
import os
import pathlib
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

from tqdm import *

from functools import wraps
import datetime
import time
import psutil
import subprocess
import cv2
from contextlib import suppress

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="video file name (or full file path) to classify")
parser.add_argument("--fastcut", default=False, action='store_true',\
                    help="cut and merge an mp4 video without re-encoding using an edit list."\
                    "Might not work on some players. see https://stackoverflow.com/a/18449609")
parser.add_argument("--window-size-divide", type=float, default=1, help="divide window size (default: 1s) by this factor")
parser.add_argument("--window-slide-divide", type=float, default=2, help="divide the window slide by this factor (default: half the window size)")
parser.add_argument("--fps", type=int, default=-1, help="frames per second of the encoded video. Lower FPS mean faster encoding (default: original)")
parser.add_argument("--crf", type=int, default=-1, help="CRF factor for h264 encoding.")
parser.add_argument("--spectrogram", default=False, action='store_true', help="print spectrogram of window_size sliding by window_slide during analysis (debubbing only)")

args = parser.parse_args()
video_path = args.filename
audio_len = None
pbar = None
tmp_folder = "tmp"
_perf = dict()
stats = None
labels = ["emh", "silence", "speech"]

cuts = []       # edits for ffmpeg to split
mergelist = []  # files for ffmpeg to merge back

def plot_spectrogram(spectrogram, ax):
  # Convert to frequencies to log scale and transpose so that the time is
  # represented in the x-axis (columns).
  log_spec = np.log(spectrogram.T)
  height = log_spec.shape[0]
  X = np.arange(16000, step=height + 1)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

def timeit(f):
    @wraps(f)
    def timed(*args, **kw):
    
        a = datetime.datetime.now()
        output = f(*args, **kw)
        b = datetime.datetime.now()
        _perf[f.__name__] = b - a
        return output
    return timed

@timeit
def convert_input(path):
    wav_path = path[:-4] + ".wav"
    print("extracting audio track...")
    cmd = ["ffmpeg", "-i", path, "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",\
          "-filter:a", "dynaudnorm", wav_path, "-y"]
    subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE).wait()
    return wav_path

@timeit
def decode_audio(audio_binary):
    audio, rate = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1), rate

def get_spectrogram(waveform, seek, window_size):
    # Padding for files with less than window_size
    if tf.shape(waveform) < window_size:
        zero_padding = tf.zeros([window_size] - tf.shape(waveform), dtype=tf.float32)
    else:
        zero_padding = tf.zeros(0, dtype=tf.float32)
  
    # Concatenate audio with padding so that all audio clips will be of the 
    # same length
    waveform = tf.cast(waveform, tf.float32)
    if tf.shape(waveform) > window_size:
        equal_length = waveform[seek:seek+window_size]
    else:
        equal_length = tf.concat([waveform, zero_padding], 0)
    # print("from:", seek, "to", seek+window_size)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)

    return spectrogram

def generate_cut(ss, to, count):
    out_name = str(count) + video_path[-4:]
    cuts.append(["ffmpeg", "-ss", ss ,"-i", video_path,  "-to", to, "-copyts"])
    if args.crf > 0:
        cuts[-1].extend(["-crf", str(args.crf)])
    if args.fps > 0:
        cuts[-1].extend(["-filter:v", "fps=fps=" + str(args.fps)])

    if args.fastcut:
        cuts[-1].extend(["-c:a", "copy", "-c:v", "copy", "-avoid_negative_ts", "1"])
    else:

        cuts[-1].extend(["-c:v", "libx264", "-crf", "23"])
    cuts[-1].extend([tmp_folder + "/" + out_name, "-y"])
    # for e in cuts[-1]:
    #     print(e + " ", end="")
    # print("")
    mergelist.append("file '" + out_name + "'")
 
@timeit
def analyze_track(model, waveform, sample_rate):
    global cuts, mergelist, pbar, audio_len, labels, stats

    # state vars for analysis loop
    lastc = 1               # last seen class
    lasts = 0              # last visited second
    lastts = "00:00:00.000" # last cut was at this timestamp
    count = 0               # number of subtitle records
    lastwf = 0      # last frame of last analyzed. for 0s --> 1s at 16000Hz would be 16000

    stats = [[0,0] for _ in range(len(labels))]
    sub = open(video_path[:-4] + ".srt", 'w', encoding = 'utf-8')  # subtitle track name
    window_size = int(sample_rate/args.window_size_divide)  # 1s by default
    window_slide = int(window_size/args.window_slide_divide)

    # slide the window of size window_size by window_slide per iteration.
    # overlap may occour.
    print("analyzing track...")
    last_i = window_slide * int(audio_len/window_slide)
    pbar = tqdm(total=last_i)
    for i in range(0, audio_len, window_slide):
        pbar.update(n=window_slide)
        spectrogram = get_spectrogram(waveform, i, window_size)
        spectrogram = tf.expand_dims(spectrogram, axis=0)

        prediction = model(spectrogram)
        cls = int(tf.math.argmax(prediction[0]))
        conf = float(tf.nn.softmax(prediction[0])[cls])

     
        # generate cut when we know the end of it (or the track is at its end)
        if cls != lastc or i == last_i:
            s = i / sample_rate
            if i == last_i:
                s += (audio_len - i) / sample_rate

            ts = "0" + str(datetime.timedelta(seconds=s))[:11]
            if len(ts) <= 8:
                ts += ".000"
            # if the window slide is overlapping the previous analyzed window
            # and prediction has changed, don't generate a new cut until we are over it
            # ...unless an "emh" is detected! [don't truncate last detected ehm]
            if labels[cls] != "emh" and i < lastwf and i < last_i:
                continue
            # generate subtitles
            record = str(count) + "\n" + lastts.replace('.',',') + " --> " + \
                     ts.replace('.',',') + "\n" + labels[lastc] + \
                     "\n[" + str(conf * 100)[:4] + "]" +"\n\n"
            count += 1
            sub.write(record)
            stats[lastc][0] += 1
            stats[lastc][1] += s - lasts
            lasts = s
            # generate cut
            if labels[lastc] == "speech":
                generate_cut(lastts, ts, count)
            lastts = ts
            lastc = cls
        # slide the right hand side of the window detection.
        # This allows to cut segments > than window size
        lastwf = i + window_size  

        if not args.spectrogram:
            continue

        img = spectrogram.numpy().T
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.putText(img, labels[cls],
                   (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("spectrogram", img)
        cv2.waitKey(1) & 0xFF
        time.sleep(0.2)
 
    sub.close()

@timeit
def cut_and_merge(out_filename):
    global pbar
    cores = int(psutil.cpu_count()/2)
    print("CUT and MERGE: running", cores, "ffmpeg simultaneous instances.")
    procs = []
    i = 0
    
    # procs pool of size <number of cores>
    for c in range(cores):
        procs.append(None)
    
    pbar = tqdm(total=len(cuts))
    # loop until all cuts are issued.
    while i < len(cuts):
        # find an empty spot on the pool and give it to the cut
        for p in range(len(procs)):
            # if the seat in the pool is empty or the occupying job has finished
            if procs[p] is None or procs[p].poll() != None:
                # if the occupying job has terminated with an error, abort everything.
                if procs[p] is not None and procs[p].poll() != 0:
                    print("there was an error with an ffmpeg process. aborting!!!")
                    exit(1)
                procs[p] = subprocess.Popen(cuts[i], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # print(procs[p], "issued. PID:", procs[p].pid)
                i += 1
                pbar.update(n=1)
                break
            time.sleep(0.01)
    
    print("\nwaiting for all processes to finish...")
    for p in procs:
        with suppress(AttributeError): p.wait()
    
    mergelist_path = tmp_folder + "/inputs.txt"
    with open(mergelist_path, 'w', encoding = 'utf-8') as f:
        for m in mergelist:
          f.write(m + "\n")

    hour = str(datetime.datetime.now().hour)
    minute = str(datetime.datetime.now().minute)
    secs = str(datetime.datetime.now().second)
    
    out_filename += "-" + hour + "-" + minute + "-" + secs + \
                   "-wsi_" + str(args.window_size_divide) + \
                   "-wsl_" + str(args.window_slide_divide) + video_path[-4:]
    
    cmd = ["ffmpeg", "-f", "concat", "-i", mergelist_path, "-c", "copy", out_filename, "-y"]
    subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE).wait()


if __name__ == '__main__':
    with suppress(FileExistsError): os.mkdir(tmp_folder)
    
    model = tf.keras.models.load_model('model')
    wav_path = convert_input(video_path)
    audio_binary = tf.io.read_file(wav_path)
    waveform, sample_rate = decode_audio(audio_binary)
    sample_rate = int(sample_rate)
    
    audio_len = len(waveform)
    
    analyze_track(model, waveform, sample_rate)
    cut_and_merge(video_path[:-4])
    
    os.remove(wav_path)

    print("\nFATTO!")
    for k, v in zip(_perf.keys(), _perf.values()):
        print(k, "ha impiegato", str(v))

    saved_time = 0

    print("")
    for i in range(len(labels)):
        if labels[i] == "speech":
            continue
        saved_time += stats[i][1]
        print("Rimosso ", stats[i][0], " ", labels[i], "(s)",
             " per un ammontare di ", str(datetime.timedelta(seconds=stats[i][1]))[:8],
             sep="")

    print("Tempo totale risparmiato:", str(datetime.timedelta(seconds=saved_time))[:8])


