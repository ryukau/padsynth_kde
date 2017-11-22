import cmath
import math
import multiprocessing
import numpy
import soundfile
import time

from functools import partial
from pathlib import Path
from pyfftw.interfaces.numpy_fft import fft, ifft


def profile(fi, bwi):
    x = fi / bwi
    return math.exp(-x * x) / bwi


def normalize(sound, amp=1.0):
    amp /= max([max(sound), -min(sound)])
    return [value * amp for value in sound]


def padsynth(samplerate, frequencies, band_width=10, random_phase=True):
    """
    PadSynth from ZynAddSubFX
    http://zynaddsubfx.sourceforge.net/doc/PADsynth/PADsynth.htm

    frequencies = [(freq, gain, phase), ...]

    profile_size_half の定数6は以下を参照。値を大きくすると遅くなるかわりに精度が上がる。
    https://en.wikipedia.org/wiki/Normal_distribution#Standard_deviation_and_coverage
    """
    table = numpy.zeros(2**16, dtype=numpy.complex)

    for freq, gain, phase in frequencies:
        band_width_hz = (math.pow(2, band_width / 1200) - 1.0) * freq
        band_width_i = band_width_hz / (2.0 * samplerate)

        sigma = math.sqrt(math.pow(band_width_i, 2.0) / (2.0 * math.pi))
        profile_size_half = max(int(6 * len(table) * sigma), 1)

        freq_i = freq / samplerate

        center = int(freq_i * len(table))
        start = max(center - profile_size_half, 0)
        end = min(center + profile_size_half, len(table))

        for index in range(start, end):
            table[index] += cmath.rect(
                gain * profile(index / len(table) - freq_i, band_width_i),
                phase)

    # table を複素数に変換。
    table[0] = 0 * 1j  # 直流を除去。
    if random_phase:
        angles = numpy.random.uniform(0, 2 * numpy.pi, len(table))
        table = table * numpy.exp(1j * angles)

    sound_ifft = ifft(table, planner_effort='FFTW_ESTIMATE', threads=1)
    sound_flat = normalize(sound_ifft.real)

    return sound_flat


def render_padsynth(index, samplerate, size, band_width, out_directory):
    numpy.random.seed()

    freq = numpy.random.uniform(100, samplerate / 2, size)
    gain = numpy.random.uniform(1e-5, 1, size)
    phase = numpy.random.uniform(0, 2.0 * numpy.pi, size)
    frequencies = [(freq[i], gain[i], phase[i]) for i in range(size)]

    sound = padsynth(samplerate, frequencies, band_width)
    soundfile.write(
        str(out_directory / Path('padsynth_{:0>4}.wav'.format(index))),
        sound,
        samplerate,
        subtype='FLOAT')


if __name__ == "__main__":
    start_time = time.perf_counter()

    out_directory = Path('render_random')
    samplerate = 48000
    size = 512
    band_width = 10
    num_render = 8

    if not out_directory.is_dir():
        if out_directory.exists():
            print(out_directory, 'already exists.')
            exit()
        out_directory.mkdir()

    pool = multiprocessing.Pool()
    func = partial(
        render_padsynth,
        samplerate=samplerate,
        size=size,
        band_width=band_width,
        out_directory=out_directory)
    pool.imap_unordered(func, [i for i in range(num_render)])
    pool.close()
    pool.join()

    end_time = time.perf_counter()
    print('Time elapsed in seconds: ', end_time - start_time)
