# audio yoobilizer

a minimalist real-time audio visualizer built with pygame, opengl, and sounddevice. designed for experimental sound installations and live visual performance.

## requirements

- macos or windows
- python 3.10 or newer
- blackhole 2ch (mac only – virtual audio driver)
- aggregate audio device (mac) or stereo mix (windows)

## installation (macos)

1. **clone the repository**

   ```bash
   git clone https://github.com/your-username/audio_yoobilizer.git
   cd audio_yoobilizer
   ```

2. **install python 3.10+**  
   you can use [pyenv](https://github.com/pyenv/pyenv) or download from [python.org](https://www.python.org/downloads/macos/)

3. **install blackhole (2ch)**  
   - download from: [https://existential.audio/blackhole/](https://existential.audio/blackhole/)
   - choose the 2-channel version
   - follow installer instructions to configure system

4. **create and activate virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **install required libraries**

   ```bash
   pip install -r requirements.txt
   ```

6. **setup aggregate audio device**
   - open the “audio midi setup” app (press `cmd + space`, type "midi")
   - click `+` in the bottom left → “create aggregate device”
   - select both your microphone/audio interface and blackhole 2ch
   - set this device as system default input/output

7. **run the visualizer**

   ```bash
   python osc.py
   ```

## installation (windows)

1. **clone the repository**

   ```cmd
   git clone https://github.com/your-username/audio_yoobilizer.git
   cd audio_yoobilizer
   ```

2. **install python 3.10+**  
   download from [python.org](https://www.python.org/downloads/windows/) and make sure to check "Add Python to PATH"

3. **enable stereo mix**
   - right-click the speaker icon → "sounds"
   - go to the "recording" tab
   - right-click → "show disabled devices"
   - enable "stereo mix"

4. **create and activate virtual environment**

   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

5. **install requirements**

   ```cmd
   pip install -r requirements.txt
   ```

6. **run the visualizer**

   ```cmd
   python osc.py
   ```

---

# audio yoobilizer

minimalistyczny wizualizator audio w czasie rzeczywistym. zbudowany na pygame, opengl i sounddevice. idealny do eksperymentalnych instalacji dźwiękowych i wizualizacji na żywo.

## wymagania

- macos lub windows
- python 3.10 lub nowszy
- blackhole 2ch (dla macos) – wirtualny sterownik audio
- skonfigurowane złożone urządzenie audio (aggregate device – macos) lub stereo mix (windows)

## instalacja (macos)

1. **sklonuj repozytorium**

   ```bash
   git clone https://github.com/twoj-login/audio_yoobilizer.git
   cd audio_yoobilizer
   ```

2. **zainstaluj pythona 3.10+**  
   możesz użyć [pyenv](https://github.com/pyenv/pyenv) albo pobrać z [python.org](https://www.python.org/downloads/macos/)

3. **zainstaluj blackhole (2 kanały)**  
   - pobierz ze strony: [https://existential.audio/blackhole/](https://existential.audio/blackhole/)
   - wybierz wersję 2-channel
   - dokończ konfigurację według instrukcji instalatora

4. **utwórz i aktywuj środowisko wirtualne**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **zainstaluj wymagane biblioteki**

   ```bash
   pip install -r requirements.txt
   ```

6. **skonfiguruj złożone urządzenie audio**
   - otwórz „konfigurator MIDI audio” (cmd + spacja → „midi”)
   - kliknij `+` → „utwórz urządzenie złożone”
   - zaznacz mikrofon/interfejs oraz blackhole
   - ustaw jako domyślne wejście/wyjście

7. **uruchom program**

   ```bash
   python osc.py
   ```

## instalacja (windows)

1. **sklonuj repozytorium**

   ```cmd
   git clone https://github.com/twoj-login/audio_yoobilizer.git
   cd audio_yoobilizer
   ```

2. **zainstaluj python 3.10+**  
   pobierz z [python.org](https://www.python.org/downloads/windows/) i zaznacz opcję "Add Python to PATH"

3. **włącz funkcję stereo mix**
   - kliknij prawym na ikonę głośnika → „dźwięki”
   - zakładka „nagrywanie”
   - kliknij PPM → „pokaż wyłączone urządzenia”
   - włącz „stereo mix”

4. **utwórz i aktywuj środowisko wirtualne**

   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

5. **zainstaluj wymagania**

   ```cmd
   pip install -r requirements.txt
   ```

6. **uruchom program**

   ```cmd
   python osc.py
   ```

---