import sounddevice as sd
import numpy as np
import queue
import threading

INPUT_DEVICE  = 2    # micro 
OUTPUT_DEVICE = 13   # CABLE Input VB-Audio
SAMPLE_RATE   = 44100
BLOCK_SIZE    = 1024
CHANNELS      = 1
NOISE_LEVEL   = 0.6

audio_queue = queue.Queue(maxsize=20)

_brown_state = 0.0

def generate_brown_noise_streaming(frames):
    global _brown_state
    white = np.random.normal(0, 0.02, frames)
    brown = np.empty(frames)
    for i in range(frames):
        _brown_state += white[i]
        _brown_state *= 0.999
        brown[i] = _brown_state
    brown /= (np.max(np.abs(brown)) + 1e-6)
    return brown

def input_callback(indata, frames, time, status):
    """
    """
    if status:
        print(f"[IN] {status}")
    try:
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        pass  


def output_callback(outdata, frames, time, status):
    """
    """
    if status:
        print(f"[OUT] {status}")
    try:
        voice = audio_queue.get_nowait()[:, 0]
    except queue.Empty:
        outdata[:] = 0
        return

    noise = generate_brown_noise_streaming(frames)
    mixed = voice + (noise * NOISE_LEVEL)
    mixed = np.clip(mixed, -1.0, 1.0)
    outdata[:, 0] = mixed


print("=" * 50)
print("  Voice Shield - Mode 2 : Noise Mask")
print("=" * 50)
print(f"  Micro source  : [{INPUT_DEVICE}] Realtek")
print(f"  Sortie        : [{OUTPUT_DEVICE}] CABLE Input")
print(f"  Niveau bruit  : {NOISE_LEVEL * 100:.0f}%")
print("  Ctrl+C pour arreter")
print("=" * 50)

try:
    stream_in = sd.InputStream(
        device=INPUT_DEVICE,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype='float32',
        channels=CHANNELS,
        callback=input_callback
    )

    stream_out = sd.OutputStream(
        device=OUTPUT_DEVICE,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype='float32',
        channels=CHANNELS,
        callback=output_callback
    )

    with stream_in, stream_out:
        print("\n[OK] Stream actif - parle dans ton micro\n")
        sd.sleep(10_000_000)

except KeyboardInterrupt:
    print("\n[OK] Arret propre du stream")
except Exception as e:
    print(f"\n[ERREUR] {e}")