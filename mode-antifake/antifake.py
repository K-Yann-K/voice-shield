import sounddevice as sd
import numpy as np
import queue
from pedalboard import Pedalboard, PitchShift


INPUT_DEVICE   = 1    # Microphone Realtek (MME)
OUTPUT_DEVICE  = 8    # CABLE Input VB-Audio (MME)
MONITOR_DEVICE = 4    # Realtek HD Audio 2nd output - ecouteurs (MME)

SAMPLE_RATE    = 44100
CHANNELS       = 1
BLOCKSIZE      = 2048

# Intensites des perturbations toutes imperceptibles a l'oreille
# Tu peux monter jusqu'a 0.01 sans entendre de difference notable
HF_NOISE_LEVEL    = 0.005   # Bruit haute frequence (4k-8kHz)
PHASE_PERTURB     = 0.003   # Perturbation de phase
BROWN_LEVEL       = 0.002   # Bruit brun residuel basse frequence


raw_queue       = queue.Queue(maxsize=20)
processed_queue = queue.Queue(maxsize=20)

# Etat du generateur de bruit brun (continu entre les blocs)
_brown_state = 0.0


def generate_hf_noise(frames):
    """
    Genere un bruit cible sur les hautes frequences (4000-8000 Hz).

    Methode : bruit blanc filtre par FFT
    1. On genere du bruit blanc (toutes frequences)
    2. On passe en domaine frequentiel (FFT)
    3. On garde uniquement les bins correspondant a 4k-8kHz
    4. On repasse en domaine temporel (FFT inverse)

    Pourquoi 4k-8kHz ?
    Les encodeurs vocaux utilisent cette bande pour capturer
    les caracteristiques du timbre et des consonnes.
    La perturber degrade la qualite de l'empreinte extraite.
    """
    # Bruit blanc
    white = np.random.normal(0, 1, frames).astype(np.float32)

    spectrum = np.fft.rfft(white)
    bin_4k = int(4000 * frames / SAMPLE_RATE)
    bin_8k = int(8000 * frames / SAMPLE_RATE)

    mask = np.zeros(len(spectrum))
    mask[bin_4k:bin_8k] = 1.0
    spectrum_filtered = spectrum * mask

    # Retour en temporel
    hf_noise = np.fft.irfft(spectrum_filtered, n=frames).astype(np.float32)

    # Normalisation
    max_val = np.max(np.abs(hf_noise))
    if max_val > 0:
        hf_noise /= max_val

    return hf_noise


def generate_brown_noise(frames):
    """
    Bruit brun continu entre les blocs (etat global _brown_state).
    Perturbe les basses frequences sans clic aux jonctions.
    """
    global _brown_state
    white = np.random.normal(0, 0.02, frames).astype(np.float32)
    brown = np.empty(frames, dtype=np.float32)

    for i in range(frames):
        _brown_state += white[i]
        _brown_state *= 0.999  # derive lente vers 0
        brown[i] = _brown_state

    max_val = np.max(np.abs(brown)) + 1e-6
    return brown / max_val


def apply_phase_perturbation(signal):
    """
    Perturbe la phase du signal de facon imperceptible.

    La phase d'un signal audio est inaudible pour l'humain
    (notre oreille est insensible aux relations de phase absolues)
    mais les reseaux de neurones d'encodage vocal en dependent
    pour reconstruire le timbre avec precision.

    Methode :
    1. FFT -> obtenir amplitude + phase de chaque bin
    2. Ajouter un bruit aleatoire a la phase (pas a l'amplitude)
    3. FFT inverse -> signal avec phase perturbee
    """
    spectrum = np.fft.rfft(signal)
    amplitude = np.abs(spectrum)
    phase     = np.angle(spectrum)

    # Perturbation aleatoire de la phase uniquement
    # PHASE_PERTURB en radians — 0.003 rad est inaudible
    phase_noise = np.random.uniform(-PHASE_PERTURB, PHASE_PERTURB, len(phase))
    phase_perturbed = phase + phase_noise

    # Reconstruction du spectre avec nouvelle phase
    spectrum_perturbed = amplitude * np.exp(1j * phase_perturbed)

    # Retour en temporel
    result = np.fft.irfft(spectrum_perturbed, n=len(signal)).astype(np.float32)
    return result


def apply_antifake(chunk):
    """
    Applique les trois couches de protection AntiFake :

    1. Perturbation de phase  — degrade l'empreinte spectrale
    2. Bruit HF (4k-8kHz)    — degrade la capture du timbre
    3. Bruit brun             — degrade les basses frequences

    Le signal reste parfaitement comprehensible pour l'humain.
    """
    frames = len(chunk)


    protected = apply_phase_perturbation(chunk)
    hf_noise = generate_hf_noise(frames)
    protected = protected + (hf_noise * HF_NOISE_LEVEL)

    brown = generate_brown_noise(frames)
    protected = protected + (brown * BROWN_LEVEL)

    return np.clip(protected, -1.0, 1.0).astype(np.float32)


def input_callback(indata, frames, time, status):
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
    1. Recupere le bloc brut
    2. Applique AntiFake
    3. Envoie vers VB-Cable + monitoring
    """
    if status:
        print(f"[CABLE] {status}")

    try:
        chunk = raw_queue.get_nowait()
    except queue.Empty:
        outdata[:, 0] = np.zeros(frames, dtype=np.float32)
        return

    protected = apply_antifake(chunk)

    out_len = min(len(protected), frames)
    outdata[:out_len, 0] = protected[:out_len]
    if out_len < frames:
        outdata[out_len:, 0] = 0

    try:
        processed_queue.put_nowait(protected.copy())
    except queue.Full:
        pass


def monitor_callback(outdata, frames, time, status):
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
print("  Voice Shield - Mode 3 : AntiFake")
print("=" * 50)
print(f"  Micro    : [{INPUT_DEVICE}] Realtek")
print(f"  Sortie   : [{OUTPUT_DEVICE}] CABLE Input")
print(f"  Monitor  : [{MONITOR_DEVICE}] Realtek HD 2nd")
print(f"  HF noise : {HF_NOISE_LEVEL * 100:.1f}%")
print(f"  Phase    : {PHASE_PERTURB:.3f} rad")
print(f"  Brown    : {BROWN_LEVEL * 100:.1f}%")
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
        print("\n[OK] Stream actif - parle normalement\n")
        print("  Ta voix est protegee contre le clonage IA")
        print("  L'interlocuteur t'entend normalement\n")
        sd.sleep(10_000_000)

except KeyboardInterrupt:
    print("\n[OK] Arret propre")
except Exception as e:
    print(f"\n[ERR] Erreur : {e}")