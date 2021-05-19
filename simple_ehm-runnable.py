#!/usr/bin/python3
# you can prepare the training dataset like so:
# ffmpeg -i full.wav -f segment -segment_time 1 -c copy out%03d.wav
# and then manually dividing the speech from ehm(s) into separate folders
import os
import io
import pathlib
import argparse
import signal

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
parser.add_argument("--generate-training-data", default=False, action='store_true', help="export extracted ehm(s) and silences as well to a separate folder. Useful for training on false positives")
parser.add_argument("--srt", default=False, action='store_true', help="generate subtitle track for easier accuracy evaluation")
parser.add_argument("--keep", nargs="+", default=["speech"], help="space separated tags to to be kept in the final video. Eg: ehm silence. Default: speech")
parser.add_argument("--output", type=str, default="", help="Output video name")
parser.add_argument("--keep-junk", default=False,action="store_true", help="keeps tmp files")

args = parser.parse_args()
video_path = args.filename
audio_len = None
pbar = None
tmp_folder = "tmp"
td_folder = "training_data"
_perf = dict()
stats = None
labels = ["ehm", "silence", "speech"]
keep = set()
trash = set()

cuts = []       # edits for ffmpeg to split
mergelist = []  # files for ffmpeg to merge back

# ctrl+c handler
def signal_handler(sig, frame):
    try:
        filename, file_extension = os.path.splitext(video_path)
        if(not args.keep_junk):
            os.remove(f'{filename}.wav')
            for file in os.listdir("tmp"):
                os.remove("tmp/"+file)

    except Exception as e:
        pass

    exit()

signal.signal(signal.SIGINT, signal_handler)


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
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",\
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
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128, pad_end=True)
    spectrogram = tf.abs(spectrogram)

    return spectrogram

def td_folder_init():
    with suppress(FileExistsError): os.mkdir(td_folder)
    removelist = [ f for f in os.listdir(td_folder) if f.endswith(".wav") ]
    for f in removelist:
        os.remove(os.path.join(td_folder, f))

def generate_tdata(ss, to, count, label):
    filename = td_folder + "/" + label + "-" + str(count) + ".wav"
    cuts.append(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", ss, "-i", video_path, "-t", "1"])
    cuts[-1].extend(["-c:a", "pcm_s16le", "-ac", "1", "-ar", "16000", "-filter:a", "dynaudnorm"])
    cuts[-1].extend([filename])

def generate_cut(ss, to, count):
    out_name = str(count) + video_path[-4:]
    cuts.append(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", ss ,"-i", video_path,  "-ss", ss,  "-to", to, "-copyts"])
    if args.crf > 0:
        cuts[-1].extend(["-crf", str(args.crf)])
    if args.fps > 0:
        cuts[-1].extend(["-filter:v", "fps=fps=" + str(args.fps)])

    if args.fastcut:
        cuts[-1].extend(["-c:a", "copy", "-c:v", "copy", "-avoid_negative_ts", "1"])
    else:
        cuts[-1].extend(["-c:v", "libx264", "-crf", "23"])
    cuts[-1].extend([tmp_folder + "/" + out_name, "-y"])
    mergelist.append("file '" + out_name + "'")
 
@timeit
def analyze_track(model, waveform, sample_rate):
    global cuts, mergelist, pbar, audio_len, labels, stats, keep, trash

    # state vars for analysis loop
    lastc = -1               # last seen class
    lasts = 0              # last visited second
    lastts = "00:00:00.000" # last cut was at this timestamp
    count = 0               # number of subtitle records
    lastwf = 0      # last frame of last analyzed. for 0s --> 1s at 16000Hz would be 16000
    stats = [[0,0] for _ in range(len(labels))]

    if args.srt:
        sub = open(video_path[:-4] + ".srt", 'w', encoding = 'utf-8')  # subtitle track name
    else:
        sub = io.StringIO()  # RAM file if no subtitle file needs to be generated

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

        if lastc == -1:
            lastc = cls
            continue
     
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
            # ...unless an undesired item is detected! 
            if labels[cls] not in trash and i < lastwf and i < last_i:
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
            if labels[lastc] in keep:
                generate_cut(lastts, ts, count)
            elif args.generate_training_data:
                generate_tdata(lastts, ts, count, labels[lastc])
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
                    print(procs[p].communicate())
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
    
    if(args.output == ""):
    	out_filename += "_" + hour + "-" + minute + "-" + secs 
    
    else:
        path = os.path.dirname(os.path.abspath(video_path))
        out_filename = os.path.abspath(path+"/"+args.output)
	    
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-i", mergelist_path, "-c", "copy", out_filename, "-y"]
    subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE).wait()


if __name__ == '__main__':
    if (not os.path.isfile(args.filename)):
       raise Exception(f"Error! {args.filename} doesn't exists.")

    with suppress(FileExistsError): os.mkdir(tmp_folder)

    if args.generate_training_data:
        td_folder_init() 

    keep = set(args.keep)
    trash = set(labels) - keep

    model = tf.keras.models.load_model('model')
    wav_path = convert_input(video_path)
    audio_binary = tf.io.read_file(wav_path)
    waveform, sample_rate = decode_audio(audio_binary)
    sample_rate = int(sample_rate)
    
    audio_len = len(waveform)
    
    analyze_track(model, waveform, sample_rate)
    cut_and_merge(video_path[:-4])
    
    if(not args.keep_junk):
        os.remove(wav_path)
        for file in os.listdir("tmp"):
            os.remove("tmp/"+file)

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


