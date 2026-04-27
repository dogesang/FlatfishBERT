"""
周期性分析模块：生成 Figure3 Panel C 所需的 periodicity_spectrum.csv
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from config import OUTPUT_DIR, SPECIES_NAMES


def sequence_to_binary(sequence: str, base: str) -> np.ndarray:
    """将DNA序列转换为指定碱基的二进制指示序列"""
    seq_upper = sequence.upper()
    return np.array([1 if b == base else 0 for b in seq_upper], dtype=np.float64)



def compute_power_spectrum(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算信号的功率谱"""
    n = len(signal)
    if n == 0:
        return np.array([]), np.array([])

    fft_result = np.fft.fft(signal)
    power = np.abs(fft_result) ** 2
    frequencies = np.fft.fftfreq(n)

    positive_mask = frequencies > 0
    return frequencies[positive_mask], power[positive_mask]



def compute_sequence_spectrum(sequence: str) -> Tuple[np.ndarray, np.ndarray]:
    """计算单条序列的综合功率谱（所有碱基平均）"""
    seq_clean = ''.join(b for b in sequence.upper() if b in 'ACGT')
    if len(seq_clean) < 10:
        return np.array([]), np.array([])

    all_powers = []
    frequencies = None

    for base in 'ACGT':
        binary = sequence_to_binary(seq_clean, base)
        freq, power = compute_power_spectrum(binary)
        if len(freq) > 0:
            if frequencies is None:
                frequencies = freq
            all_powers.append(power)

    if frequencies is None or len(all_powers) == 0:
        return np.array([]), np.array([])

    mean_power = np.mean(all_powers, axis=0)
    return frequencies, mean_power



def compute_average_spectrum(sequences: List[str], n_bins: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """计算多条序列的平均功率谱（插值到统一频率轴）"""
    freq_bins = np.linspace(0.001, 0.5, n_bins)
    accumulated_power = np.zeros(n_bins)
    count = 0

    for seq in sequences:
        freq, power = compute_sequence_spectrum(seq)
        if len(freq) == 0:
            continue
        interpolated = np.interp(freq_bins, freq, power)
        accumulated_power += interpolated
        count += 1

    if count == 0:
        return freq_bins, np.zeros(n_bins)

    mean_power = accumulated_power / count
    return freq_bins, mean_power



def compute_periodicity_by_species(
    data: Dict[str, Dict[str, List[dict]]],
    n_bins: int = 500,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """按物种和标签计算 FFT 频谱"""
    results = {}

    for species, labels_dict in data.items():
        sp_name = SPECIES_NAMES.get(species, species)
        results[species] = {}

        for label, samples in labels_dict.items():
            print(f"Computing FFT for {sp_name} - {label}: {len(samples):,} sequences")
            sequences = [s['sequence'] for s in samples]
            freq, power = compute_average_spectrum(sequences, n_bins)
            results[species][label] = (freq, power)

    return results



def compute_macro_average_spectrum(
    spectra_by_species: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """计算频谱宏平均：先按物种算，再平均"""
    labels = set()
    for sp_data in spectra_by_species.values():
        labels.update(sp_data.keys())

    results = {}
    for label in labels:
        species_powers = []
        freq = None
        for sp_data in spectra_by_species.values():
            if label in sp_data:
                f, p = sp_data[label]
                if freq is None:
                    freq = f
                species_powers.append(p)

        if species_powers and freq is not None:
            results[label] = (freq, np.mean(species_powers, axis=0))

    return results



def spectra_to_dataframe(
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]],
    spectra_by_species: Dict = None,
) -> pd.DataFrame:
    """将频谱数据转换为 DataFrame"""
    first_label = list(spectra.keys())[0]
    freq = spectra[first_label][0]

    df = pd.DataFrame({
        'frequency': freq,
        'period': 1.0 / freq,
    })

    for label, (_, power) in spectra.items():
        df[f'{label}_macro'] = power

    if spectra_by_species:
        for species, sp_data in spectra_by_species.items():
            sp_name = SPECIES_NAMES.get(species, species)
            for label, (_, power) in sp_data.items():
                df[f'{label}_{sp_name}'] = power

    return df



def save_periodicity_data(
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]],
    spectra_by_species: Dict = None,
    output_path=None,
) -> str:
    """保存 periodicity_spectrum.csv"""
    if output_path is None:
        output_path = OUTPUT_DIR / "periodicity_spectrum.csv"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = spectra_to_dataframe(spectra, spectra_by_species)
    df.to_csv(output_path, index=False)
    print(f"Saved periodicity data to: {output_path}")
    return str(output_path)
