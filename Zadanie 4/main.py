import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample # Lepsza metoda resamplingu
import sounddevice as sd
import matplotlib.pyplot as plt
import os
import time

# --- Konfiguracja ---
# Użyj istniejącego pliku WAV lub nagraj własny
# Pamiętaj, aby umieścić plik 'input.wav' w tym samym katalogu co skrypt,
# lub podaj pełną ścieżkę.
# Możesz użyć np. darmowego dźwięku z https://www.orangefreesounds.com/
DEFAULT_INPUT_WAV = "input.wav" # Przykładowy plik wejściowy
RECORD_DURATION_SECONDS = 5      # Czas nagrywania, jeśli nie ma pliku wejściowego
OUTPUT_DIR = "audio_processed_results"

# Parametry do testowania
# Typowe częstotliwości próbkowania (Hz)
SAMPLING_RATES_TO_TEST = [8000, 11025, 16000, 22050, 44100]
# Typowe głębie bitowe (bity)
# Niższe wartości (np. 2, 4) pokażą silniejszą kwantyzację
BIT_DEPTHS_TO_TEST = [4, 8, 12, 16]


# --- Funkcje pomocnicze ---

def ensure_dir(directory_path):
    """Tworzy katalog, jeśli nie istnieje."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    print(f"Wyniki będą zapisywane w katalogu: {os.path.abspath(directory_path)}")

def load_audio(filepath):
    """Wczytuje plik audio, zwraca próbki (jako float64 [-1,1]) i częstotliwość próbkowania."""
    try:
        fs, data = wav.read(filepath)
        # Normalizacja do zakresu [-1.0, 1.0] jeśli nie jest float
        if data.dtype != np.float32 and data.dtype != np.float64:
            # Sprawdzenie, czy plik jest mono czy stereo
            if len(data.shape) > 1 and data.shape[1] > 1: # Stereo lub więcej kanałów
                print(f"Plik {filepath} ma {data.shape[1]} kanałów. Używam tylko pierwszego kanału.")
                data = data[:, 0] # Bierzemy tylko pierwszy kanał

            max_val = np.iinfo(data.dtype).max
            min_val = np.iinfo(data.dtype).min
            # Normalizuj do [-1, 1]
            data = data.astype(np.float64) / max_val # Uproszczona normalizacja, dla PCM symetrycznego
                                                      # Dla niesymetrycznego uint8 może wymagać ((data/255.0)*2)-1
            # Jeśli oryginalny typ był np. uint8, zakres to 0 do 255. Po podzieleniu przez 127.5 i odjęciu 1.
            # Dla int16, zakres to -32768 do 32767. Dzielenie przez 32767.0
            if data.dtype == np.uint8: # Specjalna obsługa dla 8-bit (zwykle 0-255)
                 data = (data / 127.5) - 1.0
            else: # Dla int16, int32
                 # Dzielimy przez wartość absolutną minimum, jeśli jest większa niż maximum (np. int16)
                 # lub przez maximum, jeśli jest dodatnie.
                 # To zapewnia, że wartości będą skalowane do około [-1, 1]
                 scale = float(max(abs(min_val), abs(max_val)))
                 data = data.astype(np.float64) / scale

        print(f"Wczytano plik: {filepath}, Fs: {fs} Hz, Długość: {len(data)/fs:.2f} s, Typ danych: {data.dtype}")
        # Upewnij się, że dane są jednokanałowe (mono) dla uproszczenia
        if len(data.shape) > 1 and data.shape[1] > 1:
            print("Konwertowanie do mono przez uśrednienie kanałów.")
            data = np.mean(data, axis=1)
        return fs, data.astype(np.float64) # Zawsze zwracaj float64 dla dalszego przetwarzania
    except FileNotFoundError:
        print(f"BŁĄD: Plik {filepath} nie został znaleziony.")
        return None, None
    except Exception as e:
        print(f"BŁĄD: Nie można wczytać pliku {filepath}. {e}")
        return None, None

def save_audio(filepath, fs, data, target_bit_depth):
    """Zapisuje dane audio do pliku WAV z określoną głębią bitową."""
    # Skalowanie danych do zakresu docelowej głębi bitowej
    # i konwersja do odpowiedniego typu całkowitoliczbowego.
    # Dane wejściowe 'data' powinny być znormalizowane do [-1.0, 1.0]
    if not (-1.01 < np.min(data) < 1.01 and -1.01 < np.max(data) < 1.01):
        print(f"OSTRZEŻENIE: Dane do zapisu dla {filepath} mogą nie być w zakresie [-1,1]. Min: {np.min(data):.2f}, Max: {np.max(data):.2f}")
        data = np.clip(data, -1.0, 1.0) # Ogranicz wartości na wszelki wypadek

    if target_bit_depth == 8:
        # WAV 8-bit jest zwykle unsigned (0-255)
        scaled_data = (data * 0.5 + 0.5) * 255 # Skaluj do [0, 255]
        data_to_save = scaled_data.astype(np.uint8)
    elif target_bit_depth == 16:
        scaled_data = data * (2**15 - 1) # Skaluj do [-32767, 32767]
        data_to_save = scaled_data.astype(np.int16)
    elif target_bit_depth == 24: # scipy.io.wavfile nie wspiera bezpośrednio 24-bit
                                  # Zapiszemy jako 32-bit, ale dane będą efektywnie 24-bit
        print("OSTRZEŻENIE: scipy.io.wavfile nie wspiera bezpośrednio zapisu 24-bit. Zapisuję jako 32-bit.")
        scaled_data = data * (2**23 - 1) # Skaluj do 24-bitowego zakresu
        data_to_save = (scaled_data.astype(np.int32)) << 8 # Przesuń bity dla formatu 32-bit (często tak się robi)
                                                           # Alternatywnie, po prostu zapisz jako int32
        # data_to_save = scaled_data.astype(np.int32)
    elif target_bit_depth == 32: # float32 lub int32
        # Możemy zapisać jako float32 bezpośrednio
        # wav.write(filepath, fs, data.astype(np.float32))
        # return
        # Lub jako int32
        scaled_data = data * (2**31 - 1)
        data_to_save = scaled_data.astype(np.int32)
    else:
        print(f"Niewspierana głębia bitowa {target_bit_depth} do zapisu. Zapisuję jako 16-bit.")
        scaled_data = data * (2**15 - 1)
        data_to_save = scaled_data.astype(np.int16)

    try:
        wav.write(filepath, fs, data_to_save)
        print(f"Zapisano plik: {filepath}, Fs: {fs} Hz, Głębia: {target_bit_depth}-bit, Typ danych: {data_to_save.dtype}")
    except Exception as e:
        print(f"BŁĄD: Nie można zapisać pliku {filepath}. {e}")


def record_audio(duration_s, fs, channels=1):
    """Nagrywa audio z domyślnego mikrofonu."""
    print(f"Nagrywanie przez {duration_s} sekund z Fs={fs} Hz...")
    try:
        recording = sd.rec(int(duration_s * fs), samplerate=fs, channels=channels, dtype='float64')
        sd.wait()  # Czekaj na zakończenie nagrywania
        print("Nagrywanie zakończone.")
        if channels > 1: # Jeśli stereo, bierzemy jeden kanał
             return recording[:,0].flatten() # Zwraca jako float64 znormalizowany do [-1,1]
        return recording.flatten()
    except Exception as e:
        print(f"BŁĄD podczas nagrywania: {e}")
        print("Upewnij się, że mikrofon jest podłączony i skonfigurowany.")
        print("Dostępne urządzenia audio:")
        print(sd.query_devices())
        return None

def play_audio(fs, data, blocking=True):
    """Odtwarza dane audio."""
    # Upewnij się, że dane są w odpowiednim zakresie [-1,1] dla float lub skalowane dla int
    if data.dtype == np.float64 or data.dtype == np.float32:
        data_to_play = np.clip(data, -1.0, 1.0)
    else: # Dla typów int, sounddevice oczekuje, że są już poprawnie skalowane
        data_to_play = data

    print(f"Odtwarzanie dźwięku (Fs: {fs} Hz)...")
    try:
        sd.play(data_to_play, fs)
        if blocking:
            sd.wait()  # Czekaj na zakończenie odtwarzania
            print("Odtwarzanie zakończone.")
    except Exception as e:
        print(f"BŁĄD podczas odtwarzania: {e}")
        print("Dostępne urządzenia audio:")
        print(sd.query_devices())


# --- Przetwarzanie A/C ---

def change_sampling_rate(data, original_fs, target_fs):
    """Zmienia częstotliwość próbkowania sygnału."""
    if original_fs == target_fs:
        return data

    num_samples_original = len(data)
    duration = num_samples_original / original_fs
    num_samples_target = int(duration * target_fs)

    # Użycie resample z scipy.signal dla lepszej jakości (antyaliasing)
    resampled_data = resample(data, num_samples_target)
    print(f"Zmieniono Fs z {original_fs} Hz na {target_fs} Hz. Nowa długość: {len(resampled_data)} próbek.")
    return resampled_data


def quantize_signal(data_float, target_bit_depth):
    """Kwantyzuje sygnał (zakładając, że data_float jest w zakresie [-1.0, 1.0])."""
    # Liczba poziomów kwantyzacji
    num_levels = 2**target_bit_depth

    # Kwantyzacja symetryczna dla wartości w zakresie [-1, 1]
    # Skalujemy do [0, num_levels-1], kwantyzujemy, a potem z powrotem do [-1, 1]
    # (data_float + 1.0) / 2.0  => mapuje [-1, 1] na [0, 1]
    # * (num_levels - 1)       => mapuje [0, 1] na [0, num_levels-1]
    # np.round(...)            => kwantyzuje do najbliższej liczby całkowitej
    # / (num_levels - 1)       => mapuje z powrotem do [0, 1]
    # * 2.0 - 1.0              => mapuje z powrotem do [-1, 1]

    quantized_indices = np.round(((data_float + 1.0) / 2.0) * (num_levels - 1))
    quantized_float = (quantized_indices / (num_levels - 1)) * 2.0 - 1.0

    # Alternatywna, prostsza kwantyzacja (może być mniej dokładna dla odtworzenia zakresu)
    # scale = (2**(target_bit_depth - 1)) -1 # np. dla 8 bit to 127
    # temp_quantized = np.round(data_float * scale)
    # quantized_float = temp_quantized / scale

    print(f"Skwantyzowano sygnał do {target_bit_depth} bitów ({num_levels} poziomów).")
    return quantized_float


# --- Analiza ---
def calculate_snr(signal_original_ref, signal_processed):
    """Oblicza stosunek sygnału do szumu (SNR) w dB.
    signal_original_ref: sygnał odniesienia (najlepsza jakość).
    signal_processed: sygnał po przetworzeniu (np. kwantyzacji).
    Oba sygnały muszą mieć tę samą długość.
    """
    if len(signal_original_ref) != len(signal_processed):
        # Próba dopasowania długości, jeśli różnica jest niewielka (np. przez zaokrąglenia w resamplingu)
        min_len = min(len(signal_original_ref), len(signal_processed))
        signal_original_ref = signal_original_ref[:min_len]
        signal_processed = signal_processed[:min_len]
        if abs(len(signal_original_ref) - len(signal_processed)) > 5 : # jeśli nadal duża różnica
            print(f"OSTRZEŻENIE SNR: Znaczna różnica długości sygnałów ({len(signal_original_ref)} vs {len(signal_processed)}). Wynik SNR może być niedokładny.")
            # Można zwrócić NaN lub próbować dalej, ale z ostrzeżeniem
            # return np.nan

    noise = signal_original_ref - signal_processed
    power_signal = np.sum(signal_original_ref**2)
    power_noise = np.sum(noise**2)

    if power_noise == 0: # Idealna rekonstrukcja lub błąd
        return np.inf # Lub bardzo duża liczba
    if power_signal == 0 : # Sygnał zerowy
        return -np.inf if power_noise > 0 else 0

    snr_db = 10 * np.log10(power_signal / power_noise)
    return snr_db

def calculate_mse(signal_original_ref, signal_processed):
    """Oblicza błąd średniokwadratowy (MSE)."""
    if len(signal_original_ref) != len(signal_processed):
        min_len = min(len(signal_original_ref), len(signal_processed))
        signal_original_ref = signal_original_ref[:min_len]
        signal_processed = signal_processed[:min_len]

    error = signal_original_ref - signal_processed
    mse = np.mean(error**2)
    return mse

# --- Główna część skryptu ---
def main():
    ensure_dir(OUTPUT_DIR)

    # 1. Przygotowanie sygnału wejściowego (analogowego)
    original_fs, original_data_float = load_audio(DEFAULT_INPUT_WAV)

    if original_data_float is None:
        print(f"\nNie udało się wczytać pliku {DEFAULT_INPUT_WAV}. Próbuję nagrać dźwięk...")
        original_data_float = record_audio(RECORD_DURATION_SECONDS, 44100) # Nagraj z Fs=44100Hz
        if original_data_float is None:
            print("Nie udało się nagrać dźwięku. Przerywam.")
            return
        original_fs = 44100 # Ustaw Fs dla nagranego dźwięku
        # Zapisz nagrany oryginał dla referencji
        save_audio(os.path.join(OUTPUT_DIR, "recorded_original_44100Hz_float64.wav"),
                   original_fs, original_data_float, target_bit_depth=16) # Zapisz jako 16-bit dla łatwego odsłuchu
        print(f"Nagrano i zapisano jako 'recorded_original_44100Hz_float64.wav' (wersja float używana do przetwarzania).")

    print("\n--- Oryginalny sygnał (przed przetwarzaniem) ---")
    print(f"Fs: {original_fs} Hz, Długość: {len(original_data_float)/original_fs:.2f} s, "
          f"Min: {np.min(original_data_float):.2f}, Max: {np.max(original_data_float):.2f}")

    # Opcjonalne odtworzenie oryginalnego dźwięku
    # play_audio(original_fs, original_data_float)

    # Sygnał referencyjny dla SNR: oryginalny sygnał przy najwyższej możliwej jakości
    # (tutaj: oryginalna częstotliwość próbkowania, bez dodatkowej kwantyzacji poza tą z pliku/nagrania)
    # Jeśli oryginalny plik miał np. 16 bitów, to `original_data_float` jest jego znormalizowaną wersją.
    # Dla obliczeń SNR będziemy używać tej wersji `original_data_float` (lub jej zresamplowanej wersji)
    # jako naszego "idealnego" sygnału przed kwantyzacją do niższych głębi.

    print("\n--- Rozpoczęcie przetwarzania A/C i C/A (symulacja) ---")
    report_lines = ["Parametry;SNR (dB);MSE;Plik wyjściowy"]

    # Pętla po częstotliwościach próbkowania
    for target_fs in SAMPLING_RATES_TO_TEST:
        print(f"\n-- Przetwarzanie dla docelowej Fs: {target_fs} Hz --")

        # Krok 1: Próbkowanie (zmiana częstotliwości próbkowania)
        # Nie próbujemy zwiększać Fs, jeśli docelowa jest wyższa niż oryginalna,
        # chyba że chcemy testować algorytmy interpolacji (pomijamy dla uproszczenia).
        # Tutaj `scipy.signal.resample` poradzi sobie z upsamplingiem i downsamplingiem.
        if target_fs > original_fs:
            print(f"Docelowa Fs ({target_fs}Hz) jest wyższa niż oryginalna ({original_fs}Hz). "
                  f"Resampling (interpolacja) zostanie wykonany.")
        # else:
        #     print(f"Docelowa Fs ({target_fs}Hz) jest niższa lub równa oryginalnej ({original_fs}Hz).")

        # Resamplujemy oryginalny sygnał do docelowej częstotliwości próbkowania
        # Ten sygnał (data_resampled) będzie referencją dla SNR dla danej Fs
        data_resampled_for_reference = change_sampling_rate(original_data_float, original_fs, target_fs)

        # Pętla po głębiach bitowych
        for target_bits in BIT_DEPTHS_TO_TEST:
            print(f"\n  -- Przetwarzanie dla głębi: {target_bits} bitów (przy Fs={target_fs} Hz) --")

            # Krok 2: Kwantyzacja
            # Kwantyzujemy sygnał, który już ma docelową częstotliwość próbkowania
            data_quantized_float = quantize_signal(data_resampled_for_reference, target_bits)

            # Krok 3: Kodowanie (symulowane przez zapis do pliku WAV)
            # Nazwa pliku wyjściowego
            output_filename = f"audio_Fs{target_fs}_Bits{target_bits}.wav"
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)

            # Zapis przetworzonego sygnału (symulacja przetwornika C/A i nośnika cyfrowego)
            # data_quantized_float jest wciąż w zakresie [-1,1], save_audio przekonwertuje na int
            save_audio(output_filepath, target_fs, data_quantized_float, target_bit_depth=target_bits)

            # Opcjonalne odtworzenie przetworzonego dźwięku
            # print("  Odtwarzanie przetworzonego dźwięku...")
            # play_audio(target_fs, data_quantized_float, blocking=True)
            # time.sleep(0.5) # Krótka pauza

            # Obliczanie SNR i MSE
            # Sygnał referencyjny: data_resampled_for_reference (przed kwantyzacją do target_bits)
            # Sygnał przetworzony: data_quantized_float
            current_snr = calculate_snr(data_resampled_for_reference, data_quantized_float)
            current_mse = calculate_mse(data_resampled_for_reference, data_quantized_float)
            print(f"  SNR dla Fs={target_fs}, Bits={target_bits}: {current_snr:.2f} dB")
            print(f"  MSE dla Fs={target_fs}, Bits={target_bits}: {current_mse:.6e}")

            report_lines.append(f"Fs={target_fs}Hz Bits={target_bits};{current_snr:.2f};{current_mse:.6e};{output_filename}")

            # Wizualizacja (mały fragment dla porównania)
            if target_fs >= 8000 and (target_bits == 8 or target_bits == 16 or target_bits == 4): # Wybrane przypadki
                plt.figure(figsize=(12, 6))
                # Weźmy krótki fragment do wyświetlenia (np. 2000 próbek)
                plot_len = min(len(data_resampled_for_reference), 2000)
                time_axis = np.arange(plot_len) / target_fs

                plt.plot(time_axis, data_resampled_for_reference[:plot_len],
                         label=f"Przed kwantyzacją (Fs={target_fs}Hz)", alpha=0.7)
                plt.plot(time_axis, data_quantized_float[:plot_len],
                         label=f"Po kwantyzacji ({target_bits}-bit, Fs={target_fs}Hz)", linestyle='--')
                plt.title(f"Porównanie sygnału: Fs={target_fs} Hz, Głębia={target_bits} bitów")
                plt.xlabel("Czas [s]")
                plt.ylabel("Amplituda")
                plt.legend()
                plt.grid(True)
                plot_filename = os.path.join(OUTPUT_DIR, f"plot_Fs{target_fs}_Bits{target_bits}.png")
                plt.savefig(plot_filename)
                print(f"  Zapisano wykres: {plot_filename}")
                # plt.show() # Odkomentuj, jeśli chcesz wyświetlać wykresy na bieżąco
                plt.close()


    # Zapis raportu
    report_filepath = os.path.join(OUTPUT_DIR, "sprawozdanie_wyniki.csv")
    with open(report_filepath, 'w', encoding='utf-8') as f:
        for line in report_lines:
            f.write(line + "\n")
    print(f"\nZapisano podsumowanie wyników do: {report_filepath}")

    print("\n--- Zakończono przetwarzanie ---")
    print(f"Wszystkie przetworzone pliki audio i wykresy znajdują się w katalogu: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    # Ustawienie domyślnego urządzenia, jeśli to konieczne (opcjonalne)
    # print("Dostępne urządzenia audio:")
    # print(sd.query_devices())
    # try:
    #    sd.default.samplerate = 44100 # Ustawienie domyślnej częstotliwości próbkowania
    #    sd.default.channels = 1 # Ustawienie domyślnej liczby kanałów
    #    # sd.default.device = [numer_urządzenia_wej, numer_urządzenia_wyj] # Ustawienie urządzenia
    # except Exception as e:
    #    print(f"Nie można ustawić domyślnych parametrów sounddevice: {e}")

    main()