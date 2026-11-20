"""Assignment 5: Event-Detection"""

# author: Hannes Roedhammer
# date: 2026-11-15

# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import signal
import mat4py

sample_rate = 100  # Hz; Frequenz bzw. Abtastrate


def get_data():
    """read MatLab-File data and return it as a dictionary"""
    matlab_file = Path(__file__).resolve().parent / "data" / "s1" / "S1_A1_E1.mat"
    # Pfad in String konvertieren, da mat4py.loadmat() diesen benötigt
    file_path = str(matlab_file)
    # Pfad des aktuellen Skripts wird verwendet, um Pfad der Daten zu finden
    # print(matlab_file) # Pfad zur Kontrolle ausgeben
    return mat4py.loadmat(file_path)


def plot_stimulus(data):
    # .to_numpy() oder .values funktioniert nicht, da es sich um keinen Pandas-DataFrame handelt
    stimulus = np.array(data["stimulus"])  # Werte aus der jeweiligen Spalte bekommen
    num_samples = stimulus.shape[0]  # Anzahl der Zeilen/Messwerte
    # Zeitvektor erstellen mithilfe der Abtastrate
    # time_samples = np.arange(0, num_samples / sample_rate, 1 / sample_rate)
    time_samples = np.arange(num_samples) / sample_rate

    finger_movement_type = {
        1: "Index flexion",
        2: "Index extension",
        3: "Middle flexion",
        4: "Middle extension",
        5: "Ring flexion",
        6: "Ring extension",
        7: "Little flexion",
        8: "Little extension",
        9: "Thumb adduction",
        10: "Thumb abduction",
        11: "Thumb flexion",
        12: "Thumb extension",
    }

    # plot stimulus
    plt.plot(time_samples, stimulus)
    plt.title("Stimulus (movement repeated by the subject)")

    plt.yticks(  # Beschriftung der Bereiche Dank Copilot-Hilfe
        list(finger_movement_type.keys()),
        list(finger_movement_type.values()),
        rotation=45,
    )
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.xlabel("time / seconds")
    plt.ylabel("movement type / exercise number")

    # show the image
    plt.tight_layout()
    plt.show()


def get_total_emg(data):
    """read MatLab file and calculate total EMG by summing up all channels"""
    emg_data = np.array(data["emg"])
    """
    # Langform:
    total_emg = np.zeros(emg_data.shape[0])  # leeres Summenfeld
    # oder gleich 10 schreiben falls bekannte Spaltenanzahl
    for ii in range(emg_data.shape[1]):
        # print(emg_data[:, ii])
        total_emg += emg_data[:, ii]
    return total_emg
    """
    # kürzere Lösung
    return np.sum(emg_data, axis=1)  # Summe über alle 10 Kanäle/Spalten (axis=1)


if __name__ == "__main__":
    data = get_data()
    plot_stimulus(data)
    total_emg = get_total_emg(data)

    num_samples = total_emg.shape[0]  # Anzahl der Zeilen/Messwerte
    # Zeitvektor erstellen mithilfe der Abtastrate
    time_samples = np.arange(num_samples) / sample_rate

    """
    plt.xlabel("year")
    plt.ylabel("CO2-average")
    plt.legend()  # Legende anzeigen
    """
    # Startzeitpunkt und Endzeitpunkt des interessanten Zeitbereichsfestlegen
    t_start = 100  # in Sekunden
    t_end = 150  # in Sekunden
    # Indizes für relevanten Betrachtungszeitraum berechnen
    i_start = int(t_start * sample_rate)
    i_end = int(t_end * sample_rate)
    # Schwellwert setzen
    thresholdY = 0.2  # mV
    thresholdX = 0.5  # in Sekunden
    # ------------------
    # emg_int = np.cumulative_sum(total_emg[i_start:i_end])
    filtered = np.array(
        signal.savgol_filter(
            total_emg[i_start:i_end],
            window_length=int(thresholdX * sample_rate),
            polyorder=3,
            deriv=0,
        )
    )
    print(filtered)
    # ------------------

    # Werte mit Schwellwert vergleichen
    large_values = filtered > thresholdY
    # Änderungen feststellen (Differenzen zwischen Nachbarwerten)
    # boolean in int konvertieren mit *1
    changes = np.diff(large_values * 1)
    # Beginn und Ende der Bewegungsintervalle feststellen
    start_mov = np.where(changes == 1)[0] + i_start
    end_mov = np.where(changes == -1)[0] + i_start
    # print(start_mov / sample_rate)
    # %%
    """
    # Differenzen zwischen Start und End berechnen, um Prellungen / Bouncing zu verhindern
    start_diff = np.diff(start_mov / sample_rate) > tresholdX
    # wenn Differenz zweier Starte zu nahe beieinander liegt --> 0 gesetzt
    # Länge wieder korrigieren nach Differenzbildung und ersten Start behalten:
    start_diff = np.hstack(([1], start_diff))
    # print(start_diff)  # Kontrolle
    # nur Startwerte behalten, welche nicht zu nahe beieinander sind
    # start_mov *= start_diff  # z.B.: 0 * start_mov = 0
    start_mov = start_mov[start_diff != 0]
    print(start_mov / sample_rate)

    end_diff = np.diff(end_mov / sample_rate) > tresholdX
    end_diff = np.hstack(([1], end_diff))
    end_mov = end_mov[end_diff != 0]
    print(end_mov / sample_rate)
    """
    # %%
    # Ausgabe Plot
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(time_samples[i_start:i_end], total_emg[i_start:i_end], lw=0.5)
    axs[1].plot(time_samples[i_start:i_end], large_values, lw=0.5)
    axs[0].set_title("Total EMG")
    axs[0].set_ylabel("EMG signal / mV")
    axs[0].axhline(thresholdY, ls="dotted", color="red", label="threshold")
    axs[0].legend()
    axs[0].margins(x=0.001)
    axs[1].set_title("Thresholded EMG")
    axs[1].set_xlabel("time / seconds")

    # Bereiche der Aktivität markieren
    """
    for start, end in zip(start_mov, end_mov):
    # print(start, end) # Kontrolle
    # nur wenn Bewegung kein kurzzeitiges Artefakt
        if (end - start) / sample_rate > thresholdX:
            axs[0].axvspan(
                time_samples[start], time_samples[end], color="0.8", alpha=0.3
            )
    """
    for start, end in zip(start_mov, end_mov):
        # print(start, end) # Kontrolle
        # nur wenn Bewegung kein kurzzeitiges Artefakt
        if (end - start) / sample_rate > thresholdX:
            axs[0].axvspan(
                time_samples[start], time_samples[end], color="0.8", alpha=0.3
            )

    # Save and show the figure
    out_file = "FindingEvents.png"
    plt.savefig(out_file, dpi=300)
    print(f"Image saved to {out_file}")

    plt.show()
