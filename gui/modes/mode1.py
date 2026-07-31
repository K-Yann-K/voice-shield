import numpy as np
from pedalboard import Pedalboard, PitchShift

SAMPLE_RATE = 44100

def build_pedalboard(semitones: int) -> Pedalboard:
    """
    """
    return Pedalboard([PitchShift(semitones=semitones)])


def process(chunk: np.ndarray, board: Pedalboard) -> np.ndarray:
    """
    """
    audio_2d   = chunk.reshape(1, -1).astype(np.float32)
    shifted_2d = board(audio_2d, sample_rate=SAMPLE_RATE)
    shifted    = shifted_2d.flatten().astype(np.float32)

    # pedalboard peut retourner un tableau plus court en debut de stream
    if len(shifted) >= len(chunk):
        return np.clip(shifted[:len(chunk)], -1.0, 1.0)

    padded = np.zeros(len(chunk), dtype=np.float32)
    padded[:len(shifted)] = shifted
    return np.clip(padded, -1.0, 1.0)