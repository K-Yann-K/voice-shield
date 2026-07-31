import sounddevice as sd
from dataclasses import dataclass


@dataclass
class AudioDevice:
    """"""
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    host_api: int

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0

    def label(self) -> str:
        """
        """
        return f"[{self.index}] {self.name[:40]}"


def get_all_devices() -> list[AudioDevice]:
    """"""
    return [
        AudioDevice(
            index=i,
            name=d['name'],
            max_input_channels=d['max_input_channels'],
            max_output_channels=d['max_output_channels'],
            host_api=d['hostapi']
        )
        for i, d in enumerate(sd.query_devices())
    ]


def get_input_devices() -> list[AudioDevice]:
    """"""
    return [d for d in get_all_devices() if d.is_input]


def get_output_devices() -> list[AudioDevice]:
    """"""
    return [d for d in get_all_devices() if d.is_output]
def find_device_by_index(index: int) -> AudioDevice | None:
    """Trouve un peripherique par son index sounddevice."""
    for d in get_all_devices():
        if d.index == index:
            return d
    return None


def find_device_by_name(name: str) -> AudioDevice | None:
    """
    """
    name_lower = name.lower()
    for d in get_all_devices():
        if name_lower in d.name.lower():
            return d
    return None


def index_from_label(label: str) -> int:
    """
    Extrait l'index numerique depuis un label de dropdown.
    '[8] CABLE Input...' -> 8
    """
    return int(label.split("]")[0].replace("[", "").strip())


def get_default_devices() -> dict[str, int | None]:
    """
    """
    micro   = find_device_by_name("Microphone (Realtek")
    cable   = find_device_by_name("CABLE Input (VB-Audio Virtual C")
    monitor = find_device_by_name("Realtek HD Audio 2nd")

    return {
        "input":   micro.index   if micro   else None,
        "output":  cable.index   if cable   else None,
        "monitor": monitor.index if monitor else None,
    }