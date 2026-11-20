"""Analyse von stimulus und EMG-Daten.

Author: Judith Kraus
Programmed with AI assistance.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mat4py import loadmat
from scipy.signal import savgol_filter


FS = 100  # EMG sampling frequency in Hz
START_TIME = 100
END_TIME = 150
MIN_MOVEMENT_SECONDS = 0.2
MAX_GAP_SECONDS = 0.1


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the ten EMG channels and stimulus using mat4py."""
    data = loadmat(str(path))
    emg = np.array(data["emg"], dtype=float)
    stimulus = np.array(data["stimulus"], dtype=int).reshape(-1)
    return emg, stimulus


def detect_movements(
    total_emg: np.ndarray,
    stimulus: np.ndarray,
) -> tuple[list[tuple[int, int]], float, np.ndarray]:
    """Find movement intervals between START_TIME and END_TIME."""
    # Smooth the signal over approximately 0.2 seconds.
    smoothed = savgol_filter(total_emg, window_length=21, polyorder=2)

    # The resting signal is used to calculate an adaptive threshold.
    resting = total_emg[stimulus == 0]
    threshold = np.median(resting) + 3 * np.median(np.abs(resting - np.median(resting)))

    start = START_TIME * FS
    end = min(END_TIME * FS, len(total_emg))
    active = smoothed[start:end] > threshold
    changes = np.diff(np.r_[False, active, False].astype(int))
    starts = np.flatnonzero(changes == 1) + start
    ends = np.flatnonzero(changes == -1) + start

    movements = []
    for movement_start, movement_end in zip(starts, ends):
        if movement_end - movement_start < MIN_MOVEMENT_SECONDS * FS:
            continue
        if movements and movement_start - movements[-1][1] <= MAX_GAP_SECONDS * FS:
            movements[-1] = (movements[-1][0], movement_end)
        else:
            movements.append((movement_start, movement_end))
    return movements, threshold, smoothed


def plot_results(
    stimulus: np.ndarray,
    total_emg: np.ndarray,
    smoothed_emg: np.ndarray,
    movements: list[tuple[int, int]],
    threshold: float,
    output_path: Path,
) -> None:
    """Plot stimulus and total EMG in the selected time range."""
    times = np.arange(len(stimulus)) / FS
    selected = (times >= START_TIME) & (times <= END_TIME)

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].step(times[selected], stimulus[selected], where="post")
    axes[0].set_ylabel("stimulus")
    axes[0].set_title("S1_A1_E1: stimulus and EMG activity (100–150 s)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times[selected], total_emg[selected], alpha=0.35, label="total_emg")
    axes[1].plot(times[selected], smoothed_emg[selected], label="smoothed total_emg")
    axes[1].axhline(
        threshold, color="tab:red", linestyle="--", label=f"threshold = {threshold:.3f}"
    )
    for start, end in movements:
        axes[1].axvspan(start / FS, end / FS, color="tab:green", alpha=0.2)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("sum(abs(EMG))")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.show()


def main() -> None:
    folder = Path(__file__).resolve().parent
    data_path = folder / "s1" / "S1_A1_E1.mat"
    output_path = folder / "S1_A1_E1_analysis.png"

    emg, stimulus = load_data(data_path)
    total_emg = np.sum(np.abs(emg), axis=1)
    movements, threshold, smoothed_emg = detect_movements(total_emg, stimulus)

    print(f"Loaded {len(emg)} samples and {emg.shape[1]} EMG channels.")
    print(f"Stimulus values: {np.unique(stimulus).tolist()}")
    print(f"Movement threshold: {threshold:.3f}")
    print("Detected movement ranges:")
    for start, end in movements:
        print(f"  {start / FS:.2f} s - {end / FS:.2f} s")
    plot_results(stimulus, total_emg, smoothed_emg, movements, threshold, output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
