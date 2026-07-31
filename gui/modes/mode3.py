import numpy as np

SAMPLE_RATE = 44100

#Intensites des perturbations
HF_NOISE_LEVEL = 0.005  # Bruit haute frequence (4k-8kHz)
PHASE_PERTURB  = 0.003  # Perturbation de phase (radians)
BROWN_LEVEL    = 0.002  # Bruit brun residuel basse frequence

_brown_state: float = 0.0


def _generate_hf_noise(frames: int) -> np.ndarray:
    """
    """
    white    = np.random.normal(0, 1, frames).astype(np.float32)
    spectrum = np.fft.rfft(white)

    bin_4k = int(4000 * frames / SAMPLE_RATE)
    bin_8k = int(8000 * frames / SAMPLE_RATE)

    mask             = np.zeros(len(spectrum))
    mask[bin_4k:bin_8k] = 1.0
    filtered         = np.fft.irfft(spectrum * mask, n=frames).astype(np.float32)

    max_val = np.max(np.abs(filtered))
    return filtered / max_val if max_val > 0 else filtered


def _generate_brown_noise(frames: int) -> np.ndarray:
    """"""
    global _brown_state

    white = np.random.normal(0, 0.02, frames).astype(np.float32)
    brown = np.empty(frames, dtype=np.float32)

    for i in range(frames):
        _brown_state += white[i]
        _brown_state *= 0.999
        brown[i] = _brown_state

    max_val = np.max(np.abs(brown)) + 1e-6
    return brown / max_val
def _apply_phase_perturbation(signal: np.ndarray) -> np.ndarray:
    """
    """
    spectrum  = np.fft.rfft(signal)
    amplitude = np.abs(spectrum)
    phase     = np.angle(spectrum)

    phase += np.random.uniform(-PHASE_PERTURB, PHASE_PERTURB, len(phase))

    perturbed = amplitude * np.exp(1j * phase)
    return np.fft.irfft(perturbed, n=len(signal)).astype(np.float32)


def process(chunk: np.ndarray) -> np.ndarray:
    """
    """
    frames    = len(chunk)
    protected = _apply_phase_perturbation(chunk)
    protected = protected + _generate_hf_noise(frames) * HF_NOISE_LEVEL
    protected = protected + _generate_brown_noise(frames) * BROWN_LEVEL
    return np.clip(protected, -1.0, 1.0).astype(np.float32)


def reset_state() -> None:
    """Remet le generateur de bruit a zero — appele au stop du stream."""
    global _brown_state
    _brown_state = 0.0