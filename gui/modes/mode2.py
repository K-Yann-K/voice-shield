import numpy as np

_brown_state: float = 0.0


def _generate_brown_noise(frames: int) -> np.ndarray:
    """
    Bruit brun continu entre les blocs.

    Le bruit brun = integration du bruit blanc :
    chaque sample = precedent + petite variation aleatoire.
    Resultat : son grave et sourd car les hautes frequences
    s'annulent statistiquement, les basses s'accumulent.

    _brown_state persiste entre les appels pour eviter
    les clics aux jonctions de blocs.
    """
    global _brown_state

    white = np.random.normal(0, 0.02, frames).astype(np.float32)
    brown = np.empty(frames, dtype=np.float32)

    for i in range(frames):
        _brown_state += white[i]
        _brown_state = 0.999  # derive lente vers 0
        brown[i] = _brown_state

    max_val = np.max(np.abs(brown)) + 1e-6
    return brown / max_val


def process(chunk: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Superpose un bruit brun a la voix.
    noise_level : 0.0 = voix pure, 1.0 = voix completement masquee
    """
    frames = len(chunk)
    noise  = _generate_brown_noise(frames)
    mixed  = chunk + (noise * noise_level)
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def reset_state() -> None:
    """Remet le generateur de bruit a zero — appele au stop du stream."""
    global _brown_state
    _brown_state = 0.0