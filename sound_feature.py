import cmath
import math
import multiprocessing
import numpy
import os
import psycopg2
import random
import soundfile
import sqlite3
import time

from functools import partial
from matplotlib import pyplot as plt
from pathlib import Path
from psycopg2 import sql
from pyfftw.interfaces.numpy_fft import fft, ifft
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


## Section: Get sound features.
class SoundFeature:
    """
    cutoff と merging_range は Hz。
    """

    def __init__(self,
                 sound_file_path,
                 envelope_fitting_func,
                 interval=0.1,
                 cutoff=100,
                 spectrum_length=2**10,
                 merging_range=25):

        self.succeed = True
        try:
            sound, samplerate = soundfile.read(str(sound_file_path))
        except:
            self.succeed = False
        sound_mono = self.mix_to_mono(sound)
        sound_abs = [abs(frame) for frame in sound_mono]
        popt = self.get_curve_params(envelope_fitting_func, interval,
                                     sound_abs, samplerate)
        sound_flatten = self.flatten_sound(sound_mono, interval, sound_abs,
                                           samplerate)
        power_spectrum = self.get_power_spectrum(sound_flatten, samplerate,
                                                 cutoff)
        reduced_spectrum = self.reduce_spectrum(power_spectrum,
                                                spectrum_length, merging_range)

        self.name = str(sound_file_path.stem)
        self.group = str(sound_file_path.parent)
        self.spectrum = reduced_spectrum
        self.gain_envelope_parameters = popt.tolist()

    def mix_to_mono(self, sound):
        if isinstance(sound[0], numpy.ndarray):
            return [sum(frame) / len(frame) for frame in sound]
        return sound

    def get_curve_params(self, envelope_fitting_func, interval, sound_abs,
                         samplerate):
        indices, values = self.get_envelope_points(interval, sound_abs,
                                                   samplerate)
        indices_normalized = [index / samplerate for index in indices]
        popt, _ = curve_fit(
            envelope_fitting_func,
            indices_normalized,
            values,
            bounds=(0.0, [1.0, numpy.inf, 1.0]))
        return popt

    def get_envelope_points(self,
                            interval,
                            sound_abs,
                            samplerate,
                            flatten=False):
        interval = int(samplerate * interval)
        indices = []
        values = []
        peak_index = 0
        peak_value = 1.0
        for index, value in enumerate(sound_abs):
            if index % interval == 0:
                indices.append(peak_index)
                values.append(peak_value)
                peak_value = value
                peak_index = index
            elif peak_value < value:
                peak_value = value
                peak_index = index
        indices.append(len(sound_abs) - 1)
        if flatten:
            values.append(1.0)
        else:
            values.append(0.0)
        return (indices, values)

    def flatten_sound(self, sound_mono, interval, sound_abs, samplerate):
        indices, values = self.get_envelope_points(
            interval, sound_abs, samplerate, flatten=True)
        interp = interp1d(indices, values)
        envelope = interp([i for i in range(len(sound_mono))]).tolist()
        sound_flatten = [0.0] * len(sound_mono)
        for index, frame in enumerate(sound_mono):
            sound_flatten[index] = self.div_ceiled(frame, envelope[index])
        return sound_flatten

    def div_ceiled(self, numer, denom):
        if denom == 0:
            return 1.0
        return max(min(numer / denom, 1.0), -1.0)

    def get_power_spectrum(self, sound_flatten, samplerate, cutoff):
        raw_spectrum = fft(
            numpy.array(sound_flatten),
            planner_effort='FFTW_ESTIMATE',
        )

        samplerate_per_two = samplerate / 2
        length = int(len(raw_spectrum) / 2)
        highpass_index = math.ceil(cutoff * (length - 1) / samplerate_per_two)

        trimed_spectrum = raw_spectrum[highpass_index:int(
            len(raw_spectrum) / 2)]

        power_spectrum = numpy.absolute(trimed_spectrum)
        phase = numpy.angle(trimed_spectrum)

        samplerate_per_length = samplerate_per_two / length
        return [{
            'freq': samplerate_per_length * (index + highpass_index),
            'gain': value,
            'phase': phase[index],
        } for index, value in enumerate(power_spectrum)]

    def reduce_spectrum(self, power_spectrum, spectrum_length, merging_range):
        sorted_spectrum = sorted(
            power_spectrum, key=lambda x: x['gain'], reverse=True)

        # index = 0
        # reduced_spectrum = []
        # while len(reduced_spectrum) < spectrum_length and index < len(
        #         sorted_spectrum):
        #     if self.isUniqueFrequency(reduced_spectrum,
        #                               sorted_spectrum[index]['freq'],
        #                               merging_range):
        #         reduced_spectrum.append(sorted_spectrum[index])
        #     index += 1

        reduced_spectrum = sorted_spectrum[:spectrum_length]

        max_gain = max(reduced_spectrum, key=lambda x: x['gain'])['gain']
        for index, param in enumerate(reduced_spectrum):
            param['gain'] /= max_gain
            param['gain_index'] = index
        return reduced_spectrum

    def isUniqueFrequency(self, reduced_spectrum, frequency, merging_range):
        for elem in reduced_spectrum:
            if merging_range > abs(frequency - elem['freq']):
                return False
        return True


def envelope_fitting_func(x, a, b, c):
    return a * numpy.exp(-b * x) + c


def get_sound_feature(sound_file_path):
    return SoundFeature(sound_file_path, envelope_fitting_func)


class Table:
    def __init__(self, cursor, table_name, schema):
        self.create_table(cursor, table_name, schema)

        self.table_name = table_name
        self.schema = schema
        self.query_insert_into = self.compose_insert_into(table_name, schema)

    def create_table(self, cursor, table_name, schema, drop=True):
        """
        schema = [(name, type), (name, type), ...]
        """
        if drop:
            cursor.execute(
                sql.SQL('DROP TABLE IF EXISTS {}').format(
                    sql.Identifier(table_name)))
        schema = sql.SQL(', ').join([
            sql.SQL(' ').join([
                sql.Identifier(elem[0]),
                sql.SQL(elem[1]),
            ]) for elem in schema
        ])
        cursor.execute(
            sql.SQL('CREATE TABLE IF NOT EXISTS {table} ({schema})').format(
                table=sql.Identifier(table_name),
                schema=schema,
            ))

    def compose_insert_into(self, table_name, schema):
        # schema[0] は id。
        names = [s[0] for s in schema][1:]
        return sql.SQL('INSERT INTO {} ({}) VALUES ({}) RETURNING {}').format(
            sql.Identifier(table_name),
            sql.SQL(', ').join(map(sql.Identifier, names)),
            sql.SQL(', ').join(sql.Placeholder() * len(names)),
            sql.Identifier(schema[0][0]))

    def insert_into_one(self, cursor, data):
        cursor.execute(self.query_insert_into, data)
        return cursor.fetchone()[0]

    def insert_into_many(self, cursor, data):
        cursor.executemany(self.query_insert_into, data)
        return cursor.fetchall()


def make_sound_feature_database(sound_directory, DATABASE):
    connection = psycopg2.connect(DATABASE)
    cursor = connection.cursor()

    table_sound_group = Table(cursor, 'sound_group', [
        ('id_sound_group', 'SERIAL PRIMARY KEY'),
        ('group_name', 'TEXT NOT NULL'),
    ])
    table_sound = Table(cursor, 'sound', [
        ('id_sound', 'SERIAL PRIMARY KEY'),
        ('name', 'TEXT NOT NULL'),
        ('id_sound_group', 'INTEGER'),
        ('frequency', 'REAL[]'),
        ('gain', 'REAL[]'),
        ('phase', 'REAL[]'),
        ('gain_envelope_parameters', 'REAL[]'),
    ])

    # extract data for each sound.
    list_sound_file = list(Path(sound_directory).glob('**/*.wav'))
    pool = multiprocessing.Pool()
    for index, sound_feature in enumerate(
            pool.imap_unordered(get_sound_feature, list_sound_file), 1):
        print(index, '/', len(list_sound_file), ':', sound_feature.name)

        if not sound_feature.succeed:
            print('Could not read file :', sound_feature.name)
            continue

        cursor.execute('''
            SELECT id_sound_group
            FROM sound_group
            WHERE group_name
            LIKE %s''', (sound_feature.group, ))
        fetched = cursor.fetchall()
        id_sound_group = None
        if len(fetched) == 1:
            id_sound_group = fetched[0][0]
        elif len(fetched) == 0:
            id_sound_group = table_sound_group.insert_into_one(
                cursor, (sound_feature.group, ))
        else:
            print(
                'Database returns multiple key value. Something is going wrong.'
            )
            continue

        table_sound.insert_into_one(cursor, (
            sound_feature.name,
            id_sound_group,
            [spec['freq'] for spec in sound_feature.spectrum],
            [spec['gain'] for spec in sound_feature.spectrum],
            [spec['phase'] for spec in sound_feature.spectrum],
            sound_feature.gain_envelope_parameters,
        ))

    pool.close()
    pool.join()

    connection.commit()

    cursor.close()
    connection.close()


## Section: Make CDF lookup table.
def make_cdf(samples, size=2**10, kernel_width=0.02):
    """
    pdf: Probability density function
    cdf: Cumulative distribution function
    """
    if len(samples) == 0:
        print('Error: len(samples) == 0')
        return None

    pdf = [0] * size
    sigma2_2 = 2 * kernel_width / len(samples)  # 経験的に決めた値。
    for sample in samples:
        for index, _ in enumerate(pdf):
            # 正規分布。
            x_mu = index / len(pdf) - sample
            pdf[index] += math.exp(-x_mu * x_mu / sigma2_2)

    # pdfの最大値を1.0にする。
    max_value = max(pdf)
    pdf = [x / max_value for x in pdf]

    cdf = []
    sum_value = 0
    for value in pdf:
        sum_value += value
        cdf.append(sum_value)
    cdf = [value / sum_value for value in cdf]
    return cdf


def make_cdf_params(samples):
    min_value = min(samples)
    max_value = max(samples)
    denom = max_value - min_value
    if denom == 0:
        denom = 1.0
    cdf = make_cdf([(sample - min_value) / denom for sample in samples])
    return {'table': cdf, 'bounds': [min_value, max_value]}


def get_id_sound_group(cursor):
    cursor.execute('''
        SELECT id_sound_group
        FROM sound_group
        ''')
    return [value[0] for value in cursor.fetchall()]


def get_sound_group(cursor):
    cursor.execute('''
        SELECT id_sound_group, group_name
        FROM sound_group
        ''')
    return cursor.fetchall()


def get_gain_index_upper_bound(cursor, id_sound_group):
    cursor.execute(
        sql.SQL('''
            SELECT array_length(gain, 1)
            FROM sound
            WHERE id_sound_group={}
        ''').format(sql.Literal(id_sound_group)))
    max_gain_indices = cursor.fetchall()
    # 中央値をとる。
    a = sorted([row[0] for row in max_gain_indices])
    return a[int(len(a) / 2)]


def get_sound_data(cursor, id_sound_group, gain_index, column):
    """
    gain_index は 1 から始まる。
    """
    cursor.execute(
        sql.SQL('''
            SELECT {column}[{index}]
            FROM sound
            WHERE id_sound_group={id_sound_group}
        ''').format(
            column=sql.Identifier(column),
            index=sql.Literal(gain_index),
            id_sound_group=sql.Literal(id_sound_group),
        ))
    return [row[0] for row in cursor.fetchall()]


def get_amp_envelope_parameters(cursor, id_sound_group):
    cursor.execute(
        sql.SQL('''
            SELECT gain_envelope_parameters
            FROM sound
            WHERE id_sound_group={}
        ''').format(sql.Literal(id_sound_group)))
    params = [row[0] for row in cursor.fetchall()]
    return tuple(zip(*params))


def make_frequency_gain_cdf(gain_index, DATABASE, id_sound_group, column):
    connection = psycopg2.connect(DATABASE)
    cursor = connection.cursor()

    data = get_sound_data(cursor, id_sound_group, gain_index, column)
    if column == 'frequency':
        data = [i for i in data if not i == None]
    else:
        data = [i for i in data if not i == None]

    cursor.close()
    connection.close()

    cdf_table = make_cdf_params(data)
    cdf_table['index'] = gain_index - 1
    return cdf_table


def multiprocess_cdf_creation(id_sound_group, gain_index_ub, gain_indices,
                              column):
    cdf_tables = [None] * gain_index_ub
    cdf_bounds = [None] * gain_index_ub

    pool = multiprocessing.Pool()
    func = partial(
        make_frequency_gain_cdf,
        DATABASE=DATABASE,
        id_sound_group=id_sound_group,
        column=column)
    for _, cdf in enumerate(pool.imap_unordered(func, gain_indices)):
        index = cdf['index']
        cdf_tables[index] = cdf['table']
        cdf_bounds[index] = cdf['bounds']
    pool.close()
    pool.join()

    return (cdf_tables, cdf_bounds)


def create_cdf_tables(DATABASE):
    print('Creating CDF tables.')

    connection = psycopg2.connect(DATABASE)
    cursor = connection.cursor()

    # some_bounds = [[min, max], [min, max], ...]
    resynth_cdf = Table(cursor, 'resynth_cdf', [
        ('id_cdf', 'SERIAL PRIMARY KEY'),
        ('id_sound_group', 'INTEGER'),
        ('cdf_frequency', 'REAL[][]'),
        ('cdf_frequency_bounds', 'REAL[][2]'),
        ('cdf_gain', 'REAL[][]'),
        ('cdf_gain_bounds', 'REAL[][2]'),
        ('cdf_phase', 'REAL[][]'),
        ('cdf_phase_bounds', 'REAL[][2]'),
        ('cdf_gain_envelope', 'REAL[][]'),
        ('cdf_gain_envelope_bounds', 'REAL[][2]'),
    ])

    list_id_sound_group = get_id_sound_group(cursor)
    for id_sound_group in list_id_sound_group:
        print('Processing id_sound_group', id_sound_group)

        gain_index_ub = get_gain_index_upper_bound(cursor, id_sound_group)
        gain_indices = [i for i in range(1, gain_index_ub + 1)]

        cdf_dict = {}
        columns = ['frequency', 'gain', 'phase']
        for column in columns:
            cdf_dict[column] = multiprocess_cdf_creation(
                id_sound_group, gain_index_ub, gain_indices, column)

        cdf_gain_env = []
        cdf_gain_env_bounds = []

        gain_env_params = get_amp_envelope_parameters(cursor, id_sound_group)
        for param in gain_env_params:
            cdf_param = make_cdf_params(param)
            cdf_gain_env.append(cdf_param['table'])
            cdf_gain_env_bounds.append(cdf_param['bounds'])

        resynth_cdf.insert_into_one(cursor, (
            id_sound_group,
            cdf_dict['frequency'][0],
            cdf_dict['frequency'][1],
            cdf_dict['gain'][0],
            cdf_dict['gain'][1],
            cdf_dict['phase'][0],
            cdf_dict['phase'][1],
            cdf_gain_env,
            cdf_gain_env_bounds,
        ))

    connection.commit()

    cursor.close()
    connection.close()


## Section: Resynthesize.
class CDF:
    def __init__(self, cdf_table, min_random=0.0, max_random=1.0):
        self.cdf = [0.0] + cdf_table
        self.min = min_random
        self.diff_min_max = max_random - min_random

    def map(self):
        """
        [0-1]の範囲でそれなりに分布に従った値を返す。
        random()から使う。
        """
        value = random.random()
        cdf_length = len(self.cdf)
        if cdf_length <= 1:
            return 0.0
        index = int(cdf_length / 2 - 1)
        half = int(cdf_length / 4)

        while half > 0:
            if value > self.cdf[index]:
                index += half
            elif value < self.cdf[index]:
                index -= half
            else:
                if value == 1.0:
                    while value <= self.cdf[index]:
                        index -= 1
                    return index + 1
                elif value == 0.0:
                    while value >= self.cdf[index]:
                        index += 1
                    return index - 1
                return index
            half = int(half / 2)

        # self.cdf の中で [..., 0, 0, 0, 0, ...] のように 0 が続いている場合の対策。
        while value < self.cdf[index]:
            index -= 1

        # はみだした値を補間。
        interp = (value - self.cdf[index]) / (
            self.cdf[index + 1] - self.cdf[index])
        return (index + interp) / cdf_length

    def random(self):
        return self.min + self.diff_min_max * self.map()


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


def load_cdf(DATABASE, id_sound_group):
    connection = psycopg2.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute(
        sql.SQL('''
        SELECT
            cdf_frequency,
            cdf_frequency_bounds,
            cdf_gain,
            cdf_gain_bounds,
            cdf_phase,
            cdf_phase_bounds,
            cdf_gain_envelope,
            cdf_gain_envelope_bounds
        FROM resynth_cdf
        WHERE id_sound_group={}''').format(sql.Literal(id_sound_group)))
    fetched = cursor.fetchone()

    cursor.close()
    connection.close()

    # ここひどい。
    cdf_freq = fetched[0]
    cdf_freq_bounds = fetched[1]
    cdf_gain = fetched[2]
    cdf_gain_bounds = fetched[3]
    cdf_phase = fetched[4]
    cdf_phase_bounds = fetched[5]
    cdf_gain_envelope = fetched[6]
    cdf_gain_envelope_bounds = fetched[7]

    cdf = {
        'spectrum': [(
            CDF(
                cdf_freq[i],
                cdf_freq_bounds[i][0],
                cdf_freq_bounds[i][1],
            ),
            CDF(
                cdf_gain[i],
                cdf_gain_bounds[i][0],
                cdf_gain_bounds[i][1],
            ),
            CDF(
                cdf_phase[i],
                cdf_phase_bounds[i][0],
                cdf_phase_bounds[i][1],
            ),
        ) for i in range(len(cdf_freq))],
        'gain_envelope': [
            CDF(
                cdf_gain_envelope[i],
                cdf_gain_envelope_bounds[i][0],
                cdf_gain_envelope_bounds[i][1],
            ) for i in range(len(cdf_gain_envelope))
        ],
    }

    return cdf


def multiprocess_resynth(index, cdf, samplerate, band_width, group_name):
    frequencies = []
    for _ in range(1):
        frequencies += [(
            cdf_freq.random(),
            cdf_gain.random(),
            cdf_phase.random(),
        ) for cdf_freq, cdf_gain, cdf_phase in cdf['spectrum']]

    sound_flat = numpy.array(padsynth(samplerate, frequencies, band_width))
    sound_flat_phase = numpy.array(
        padsynth(samplerate, frequencies, band_width, False))

    gain_envelope = envelope_fitting_func(
        numpy.array([i / samplerate for i in range(len(sound_flat))]),
        cdf['gain_envelope'][0].random(),
        cdf['gain_envelope'][1].random(),
        cdf['gain_envelope'][2].random(),
    )

    sound_resynthed = numpy.multiply(sound_flat, gain_envelope)

    soundfile.write(
        'render/flat_{:0>4}-{:0>4}.wav'.format(group_name, index),
        sound_flat,
        samplerate,
        subtype='FLOAT')
    soundfile.write(
        'render/flat_phase_{:0>4}-{:0>4}.wav'.format(group_name, index),
        sound_flat_phase,
        samplerate,
        subtype='FLOAT')
    soundfile.write(
        'render/out_{:0>4}-{:0>4}.wav'.format(group_name, index),
        sound_resynthed,
        samplerate,
        subtype='FLOAT')


def resynth(DATABASE, band_width=10):
    connection = psycopg2.connect(DATABASE)
    cursor = connection.cursor()

    sound_groups = get_sound_group(cursor)

    cursor.close()
    connection.close()

    num_render = 8
    samplerate = 48000

    for id_sound_group, group_name in sound_groups:
        print('Group', id_sound_group, ': Loading CDF.')
        cdf = load_cdf(DATABASE, id_sound_group)

        print('Group', id_sound_group, ': Rendering.')
        pool = multiprocessing.Pool()
        func = partial(
            multiprocess_resynth,
            cdf=cdf,
            samplerate=samplerate,
            band_width=band_width,
            group_name=str(Path(group_name).name))
        pool.map(func, [i for i in range(num_render)])
        pool.close()
        pool.join()


if __name__ == "__main__":
    user = os.getlogin()
    DATABASE = 'dbname=' + user + ' user=' + user

    start_time = time.perf_counter()

    make_sound_feature_database('sample', DATABASE)
    create_cdf_tables(DATABASE)
    resynth(DATABASE, 10)

    end_time = time.perf_counter()
    print('Time elapsed in seconds: ', end_time - start_time)
