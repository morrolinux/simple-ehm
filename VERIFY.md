# Verifica binary – essere sicuri che non ci sia roba malevola

I binary in `dist/` sono compilati dai sorgenti di questa cartella con
PyInstaller (vedi `build_windows.ps1` / `build_linux.sh`). Per fidarti,
non fidarti: verifica.

## 1. Verifica hash (integrità)

Ogni release pubblica `SHA256SUMS.txt` con gli hash dei file.
Confronta il file che hai scaricato:

Windows (PowerShell):

```powershell
Get-FileHash .\dist\simple-ehm.exe -Algorithm SHA256
Get-FileHash .\dist\simple-ehm-gui.exe -Algorithm SHA256
Get-Content .\SHA256SUMS.txt
```

Linux:

```bash
sha256sum dist/simple-ehm dist/simple-ehm-gui
cat SHA256SUMS.txt
```

Gli hash devono coincidere carattere per carattere. Se differiscono anche di
una cifra: **non eseguire il file**.

## 2. Da dove vengono gli hash? (catena di fiducia)

1. L'autore compila dai sorgenti (`git log` pubblico) e pubblica gli hash.
2. Chiunque può **ricompilare dai sorgenti** e confrontare: stesso codice →
   stesso comportamento. Istruzioni in `README.md` (sezione Binaries).
3. Il `.deb` si ispeziona senza installare:
   `dpkg-deb --contents simple-ehm_*_amd64.deb`.

Nessun passaggio richiede di fidarsi di un singolo file: codice pubblico +
build riproducibile + hash pubblicati.

## 3. Controllo antivirus / sandbox (consigliato per ogni exe da internet)

- Windows Defender: tasto destro sul file → *Analizza con Microsoft Defender*.
- Online (multi-motore): https://www.virustotal.com (carica l'exe;
  qualche falso positivo euristico su PyInstaller è normale e documentato,
  ma 0/votazioni malevole gravi è il segnale atteso).
- Prima esecuzione in sandbox: `Windows Sandbox` oppure VM.

Nota: gli exe **non sono firmati** con certificato (SmartScreen di Windows
mostrerà "Autore sconosciuto" → *Ulteriori informazioni → Esegui comunque*
solo dopo le verifiche sopra).

## 4. Cosa fa il programma in rete? (poco, e solo questo)

- All'avvio: check release GitHub (`api.github.com`, repo `SIMPLE_EHM_REPO`)
  per versione programma e `manifest.json` modelli.
- Download modello `model.keras` **solo se manca o è più vecchio** del remoto
  (verifica dimensione/sha256 da manifest). Mai eseguibili, solo pesi.
- Disattivabile: `--no-download` (e `SIMPLE_EHM_MODEL_URL` per mirror proprio).
- Niente telemetria, niente upload.

## 5. Ricompila tu (massima garanzia)

```powershell
pip install -r requirements.txt
pip install pyinstaller
pyinstaller simple-ehm-cli.spec
pyinstaller simple-ehm-gui.spec
```

Confronta gli hash del tuo build con quelli pubblicati (a parità di
versioni di Python/librerie il comportamento è identico; gli hash byte per
byte possono variare tra macchine diverse – conta il codice sorgente).
