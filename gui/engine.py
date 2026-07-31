import queue
import threading
import numpy as np
import sounddevice as sd
from pedalboard import Pedalboard
from gui.modes import mode1, mode2, mode3


class AudioEngine:
    """
    """

    SAMPLE_RATE = 44100
    BLOCKSIZE   = 2048
    CHANNELS    = 1

    def __init__(self):
        self.running: bool         = False
        self.mode: int | None      = None
        self.pitch_shift: int      = 4
        self.noise_level: float    = 0.6

        self.input_device: int | None   = None
        self.output_device: int | None  = None
        self.monitor_device: int | None = None

        self.raw_queue: queue.Queue       = queue.Queue(maxsize=20)
        self.processed_queue: queue.Queue = queue.Queue(maxsize=20)

        self.board: Pedalboard | None = None
        self._thread: threading.Thread | None = None


    def start( self, mode: int, input_device: int, output_device: int, monitor_device: int ) -> None:
        """
        """
        self.mode           = mode
        self.input_device   = input_device
        self.output_device  = output_device
        self.monitor_device = monitor_device
        self.running        = True

        # Reconstruction du pedalboard si mode 1
        if mode == 1:
            self.board = mode1.build_pedalboard(self.pitch_shift)

        # Reset des etats internes des generateurs de bruit
        mode2.reset_state()
        mode3.reset_state()

        # Vidage des queues pour eviter des residus du stream precedent
        self._flush_queues()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Arrete proprement le stream en cours."""
        self.running = False

    def update_pitch(self, semitones: int) -> None:
        """
        Met a jour le pitch shift a chaud.
        Reconstruit le pedalboard si le stream est actif.
        """
        self.pitch_shift = semitones
        if self.running and self.mode == 1:
            self.board = mode1.build_pedalboard(semitones)

    def update_noise(self, level: float) -> None:
        """Met a jour le niveau de bruit a chaud (Mode 2)."""
        self.noise_level = level


    def _input_callback( self, indata: np.ndarray, frames: int,  ts, status ) -> None:
        """
        """
        if status:
            print(f"[IN] {status}")

        try:
            self.raw_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def _output_callback( self, outdata: np.ndarray, frames: int, ts, status ) -> None:
        """
        """
        if status:
            print(f"[OUT] {status}")

        try:
            chunk = self.raw_queue.get_nowait()
        except queue.Empty:
            outdata[:, 0] = np.zeros(frames, dtype=np.float32)
            return

        processed = self._process(chunk)

        out_len = min(len(processed), frames)
        outdata[:out_len, 0] = processed[:out_len]
        if out_len < frames:
            outdata[out_len:, 0] = 0

        try:
            self.processed_queue.put_nowait(processed.copy())
        except queue.Full:
            pass


    def _monitor_callback( self, outdata: np.ndarray, frames: int, ts, status) -> None:
        """
        """
        if status:
            print(f"[MON] {status}")

        try:
            chunk   = self.processed_queue.get_nowait()
            out_len = min(len(chunk), frames)
            outdata[:out_len, 0] = chunk[:out_len]
            if out_len < frames:
                outdata[out_len:, 0] = 0
        except queue.Empty:
            outdata[:, 0] = np.zeros(frames, dtype=np.float32)


    def _process(self, chunk: np.ndarray) -> np.ndarray:
        """
        Route le bloc audio vers le bon module selon le mode actif.
        """
        if self.mode == 1 and self.board is not None:
            return mode1.process(chunk, self.board)
        if self.mode == 2:
            return mode2.process(chunk, self.noise_level)
        if self.mode == 3:
            return mode3.process(chunk)
        return chunk  # mode inconnu -> bypass


    def _run(self) -> None:
        """
        """
        try:
            with sd.InputStream( samplerate=self.SAMPLE_RATE, blocksize=self.BLOCKSIZE, dtype='float32',
                device=self.input_device,
                channels=self.CHANNELS,
                callback=self._input_callback
            ), sd.OutputStream( samplerate=self.SAMPLE_RATE, blocksize=self.BLOCKSIZE, dtype='float32',
                device=self.output_device,
                channels=self.CHANNELS,
                callback=self._output_callback
            ), sd.OutputStream( samplerate=self.SAMPLE_RATE, blocksize=self.BLOCKSIZE, dtype='float32',
                device=self.monitor_device,
                channels=self.CHANNELS,
                callback=self._monitor_callback
            ):
                while self.running:
                    sd.sleep(100)

        except Exception as e:
            print(f"[ERR AudioEngine] {e}")
            self.running = False


    def _flush_queues(self) -> None:
        """Vide les deux queues pour repartir d'un etat propre."""
        for q in (self.raw_queue, self.processed_queue):
            while not q.empty():
                q.get()