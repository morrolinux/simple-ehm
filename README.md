# simple-ehm
A simple tool for a simple task: remove filler sounds ("ehm") from pre-recorded speeches. AI powered.
Istruzioni in italiano in fondo al documento.

# Usage
Basic invokation should be enough:
`./simple_emh-runnable.py /path/to/video/file`
This will generate a subtilte track (`.srt`) for debugging and the output video in the same folder as the original file.

For more info read the help:
`./simple_emh-runnable.py --help`

## Arguments:
```text
    filename                    video file name (or full file path) to classify
    --fastcut                   cut and merge an mp4 video without re-encoding using an edit list.
    --window-size-divide        divide window size (default: 1s) by this factor
    --window-slide-divide       divide the window slide by this factor (default: half the window size)
    --fps                       frames per second of the encoded video. Lower FPS mean faster encoding (default: original)
    --crf                       CRF factor for h264 encoding.
    --spectrogram               print spectrogram of window_size sliding by window_slide during analysis (debubbing only)
    --generate-training-data    export extracted ehm(s) and silences as well to a separate folder. Useful for training on false positives)
    --srt                       generate subtitle track for easier accuracy evaluation
    --keep                      space separated tags to to be kept in the final video. Eg: ehm silence. Default: speech
    --output                    output video name
    --keep-junk                 keeps tmp files
```

# Install
This script requires `ffmpeg` as dependency.

## Linux
- Install `ffpmeg` with your package manager (Ex: `sudo apt install ffmpeg` for debian-based distros).
- In your installation folder:<br>
  ```bash
  $ git clone https://github.com/morrolinux/simple-ehm
  $ cd simple-ehm
  $ pip3 install -r requirements.txt
  ```

## Windows
- Follow the [guide about ffmpeg installation](https://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/)
- In your installation folder:<br>
  ```bash
  $ git clone https://github.com/morrolinux/simple-ehm
  $ cd simple-ehm
  $ pip3 install -r requirements.txt
  ```

**NB**: You may need to remove the *MAX_PATH limitation* in order to install dependencies, follow [this official guide](https://docs.python.org/3.7/using/windows.html#removing-the-max-path-limitation).

# Contributing to the model
There are two ways you can contribute to the model:

## Contribute to the dataset
By sending me at least 30 1-second long WAV pcm_s16le mono 16kHz clips for each class (silence, speech, ehm)  [easy]
- You can convert your clips to the right format with ffmpeg: `ffmpeg -i input-file -c:a pcm_s16le -ac 1 -ar 16000 -filter:a "dynaudnorm" output.wav`
- You can extract ehm(s) and silences **along with erroneously classified sounds** (false positives) by passing `--generate-training-data` as an invocation parameter. You can then use the latter to improve your training set!

## Contribute to the training
- By implementing transfer training logic on this model's python notebook
- By retraining the current model with your dataset and make a PR with the updated one


# ITA

# simple-ehm
Un semplice strumento per un semplice compito: rimuovere gli "ehm" (suoni di riempimento) da discorsi pre-registrati. 

# Utilizzo
L'invocazione base dovrebbe essere sufficiente:
`./simple_emh-runnable.py /percorso/al/file/video`
Questo genererò una traccia di sottotitoli (`.srt`) per fini diagnostici e il video tagliato nella stessa cartella del file originale.

Per maggiori informazioni sui parametri accettati, leggi la guida:
`./simple_emh-runnable.py --help`

## Argomenti:
```text
    filename                    video file name (or full file path) to classify
    --fastcut                   cut and merge an mp4 video without re-encoding using an edit list.
    --window-size-divide        divide window size (default: 1s) by this factor
    --window-slide-divide       divide the window slide by this factor (default: half the window size)
    --fps                       frames per second of the encoded video. Lower FPS mean faster encoding (default: original)
    --crf                       CRF factor for h264 encoding.
    --spectrogram               print spectrogram of window_size sliding by window_slide during analysis (debubbing only)
    --generate-training-data    export extracted ehm(s) and silences as well to a separate folder. Useful for training on false positives)
    --srt                       generate subtitle track for easier accuracy evaluation
    --keep                      space separated tags to to be kept in the final video. Eg: ehm silence. Default: speech
    --output                    output video name
    --keep-junk                 keeps tmp files
```

# Installazione
Per utilizzare il software è necessario installare `ffmpeg`.

## Linux
- Installare `ffpmeg` mediante il gestore pacchetti della vostra distribuzione (Es: `sudo apt install ffmpeg` per debian derivate).
- Nella cartella in cui si desidera installare:<br>
  `git clone https://github.com/morrolinux/simple-ehm` <br>
  `cd simple-ehm`<br>
  `pip3 install -r requirements.txt`
## Windows
- Seguire la [guida di installazione di ffmpeg](https://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/)
- Nella cartella in cui si desidera installare:<br>
  ```bash
  $ git clone https://github.com/morrolinux/simple-ehm
  $ cd simple-ehm
  $ pip3 install -r requirements.txt
  ```

**NB**: Per installare le dipendenze potrebbe essere necessario rimuovere il *MAX_PATH limitation*, seguire [questa guida ufficiale](https://docs.python.org/3.7/using/windows.html#removing-the-max-path-limitation) (aprire l'editor registri e navigare al suo interno per trovare l'etichetta).

# Contribuire al modello
Ci sono due modi in cui puoi contribuire al modello:

## Contribuisci al dataset
Inviandomi almeno 30 clip in formato WAV (pcm_s16le) mono con campionamento a 16kHz per ciascuna classe (silenzio, parlato, ehm)  [facile]
- Puoi convertire le tue clip nel formato corretto con ffmpeg: `ffmpeg -i input-file -c:a pcm_s16le -ac 1 -ar 16000 -filter:a "dynaudnorm" output.wav`
- Puoi estrarre gli ehm(s) e i silenzi **anche quelli classificati erroneamente** (falsi positivi) passando `--generate-training-data` come parametro di invocazione. Puoi usare le clip classificate erroneamente per migliorare il tuo training set!

## Contribuisci al training
- Implementando la logica di transfer training sul notebook python di questo modello, e
- Eseguendo il retraining della rete esistente con il tuo dataset ed inviandomi il modello aggiornato.