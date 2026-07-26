import sounddevice as sd
import numpy as np
import queue
from pedalboard import Pedalboard, PitchShift


INPUT_DEVICE   = 1    # Microphone Realtek (MME)
OUTPUT_DEVICE  = 8    # CABLE Input VB-Audio (MME)
MONITOR_DEVICE = 4    # Realtek HD Audio 2nd output - ecouteurs (MME)

SAMPLE_RATE    = 44100
CHANNELS       = 1
BLOCKSIZE      = 2048  # Plus grand bloc = meilleure qualite rubberband

PITCH_SHIFT    = -1    # Demi-tons : +4 aigu, -4 grave, +12 octave


# Une seule chaine :
# input -> raw_queue -> traitement -> processed_queue -> cable + monitor
raw_queue       = queue.Queue(maxsize=20)
processed_queue = queue.Queue(maxsize=20)


board = Pedalboard([PitchShift(semitones=PITCH_SHIFT)])

def process_pitch(chunk):
    """
    """
    audio_2d = chunk.reshape(1, -1).astype(np.float32)
    shifted_2d = board(audio_2d, sample_rate=SAMPLE_RATE)
    shifted = shifted_2d.flatten().astype(np.float32)

    if len(shifted) >= len(chunk):
        return np.clip(shifted[:len(chunk)], -1.0, 1.0)
    else:
        padded = np.zeros(len(chunk), dtype=np.float32)
        padded[:len(shifted)] = shifted
        return np.clip(padded, -1.0, 1.0)


def input_callback(indata, frames, time, status):
    """
    """
    if status:
        print(f"[IN] {status}")

    chunk = indata[:, 0].copy()

    try:
        raw_queue.put_nowait(chunk)
    except queue.Full:
        pass


def cable_callback(outdata, frames, time, status):
    """
    Sortie VB-Cable :
    1. Recupere bloc brut depuis raw_queue
    2. Applique pitch shift via Rubber Band
    3. Envoie vers VB-Cable
    4. Copie le resultat dans processed_queue pour le monitoring
    """
    if status:
        print(f"[CABLE] {status}")

    try:
        chunk = raw_queue.get_nowait()
    except queue.Empty:
        outdata[:, 0] = np.zeros(frames, dtype=np.float32)
        return

    # Traitement pitch shift
    processed = process_pitch(chunk)
    out_len = min(len(processed), frames)
    outdata[:out_len, 0] = processed[:out_len]

    if out_len < frames:
        outdata[out_len:, 0] = 0
    try:
        processed_queue.put_nowait(processed.copy())
    except queue.Full:
        pass


def monitor_callback(outdata, frames, time, status):
    """
    """
    if status:
        print(f"[MON] {status}")

    try:
        chunk = processed_queue.get_nowait()
        out_len = min(len(chunk), frames)
        outdata[:out_len, 0] = chunk[:out_len]
        if out_len < frames:
            outdata[out_len:, 0] = 0
    except queue.Empty:
        outdata[:, 0] = np.zeros(frames, dtype=np.float32)


print("=" * 50)
print("  Voice Shield - Mode 1 : Voice Transform")
print("=" * 50)
print(f"  Micro    : [{INPUT_DEVICE}] Realtek")
print(f"  Sortie   : [{OUTPUT_DEVICE}] CABLE Input")
print(f"  Monitor  : [{MONITOR_DEVICE}] Realtek HD 2nd")
print(f"  Pitch    : {'+' if PITCH_SHIFT > 0 else ''}{PITCH_SHIFT} demi-tons")
print("  Ctrl+C pour arreter")
print("=" * 50)

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        dtype='float32',
        device=INPUT_DEVICE,
        channels=CHANNELS,
        callback=input_callback
    ), sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        dtype='float32',
        device=OUTPUT_DEVICE,
        channels=CHANNELS,
        callback=cable_callback
    ), sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        dtype='float32',
        device=MONITOR_DEVICE,
        channels=CHANNELS,
        callback=monitor_callback
    ):
        sd.sleep(10_000_000)

except KeyboardInterrupt:
    print("\n[OK] Arret propre")
except Exception as e:
    print(f"\n[ERR] Erreur : {e}")