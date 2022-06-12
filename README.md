# simple-ehm
A simple tool for a simple task: remove filler sounds ("ehm") from pre-recorded speeches. AI powered.
Istruzioni in italiano in fondo al documento.

# Usage
Basic invokation should be enough:
`./simple_emh-runnable.py /path/to/video/file`
This will generate a subtitle track (`.srt`) for debugging and the output video in the same folder as the original file.

For more info read the help:
`./simple_emh-runnable.py --help`

You can also run simple-ehm in a dockerized environment. Simply use the Dockerfile to build the image then, instead of
using `./simple_ehm-runnable.py` use `./convert.sh`

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

Puoi anche utilizzare simple-ehm in un ambiente dockerizzato, per fare ciò, dove useresti `./simple_ehm-runnable.py`
utilizza invece `./convert.sh` (N.B. per usare `./convert.sh` i file devono essere spostati prima in questa cartella)

# Contribuire al modello
Ci sono due modi in cui puoi contribuire al modello:

## Contribuisci al dataset
Inviandomi almeno 30 clip in formato WAV (pcm_s16le) mono con campionamento a 16kHz per ciascuna classe (silenzio, parlato, ehm)  [facile]
- Puoi convertire le tue clip nel formato corretto con ffmpeg: `ffmpeg -i input-file -c:a pcm_s16le -ac 1 -ar 16000 -filter:a "dynaudnorm" output.wav`
- Puoi estrarre gli ehm(s) e i silenzi **anche quelli classificati erroneamente** (falsi positivi) passando `--generate-training-data` come parametro di invocazione. Puoi usare le clip classificate erroneamente per migliorare il tuo training set!

## Contribuisci al training
- Implementando la logica di transfer training sul notebook python di questo modello, e
- Eseguendo il retraining della rete esistente con il tuo dataset ed inviandomi il modello aggiornato.
