# KDEとPADsynthを用いた加算合成シンセサイザ
適当なサンプルを入力すると似たような周波数成分の音を合成するプログラムです。

もう少し詳しい仕組みについては [doc/ride.org](./doc/ride.org) を参照してください。

- [PADsynth](http://zynaddsubfx.sourceforge.net/doc/PADsynth/PADsynth.htm)
- [Kernel density estimation (KDE)](https://en.wikipedia.org/wiki/Kernel_density_estimation)

# どんな音が出るの?
こんな音がでます。

- [KDEを用いた合成音](./render)
- [ランダムな合成音](./render_random)

# 使い方
## padsynth.py
乱数でPADsynthのパラメータを生成して音を合成します。

以下のコマンドで実行します。

```
python3 padsynth.py
```

## sound\_feature.py
sound\_feature.pyはKDEでサンプルから抽出した特徴に基づいてPADsynthのパラメータを生成して音を合成します。

以下のpython3ライブラリが必要です。

- [SciPy](https://docs.scipy.org/doc/scipy/reference/)
- [NumPy](http://www.numpy.org/)
- [PySoundFile](http://pysoundfile.readthedocs.io/en/0.9.0/)
- [psycopg2](http://initd.org/psycopg/)
- [pyFFTW](http://hgomersall.github.io/pyFFTW/)

このプログラムはPostgreSQLを利用します。プログラムを実行するユーザ名と同じ名前のデータベースが必要です。

以下はFedora 27でPostgreSQLをセットアップする手順です。 user\_name と db\_name は同じにしてください。

```bash
sudo dnf install postgresql postgresql-server postgres-contrib
sudo systemctl enable postgresql
sudo postgresql-setup --initdb --unit postgresql
sudo sytemctl start postgresql
sudo -u postgres -i createuser user_name
sudo -u postgres -i createdb --owner=user_name db_name
```

sound\_feature.py と同じディレクトリに sample というディレクトリを作成して、その中にサンプルを配置してください。サンプルの形式はPySoundFileで読み込めるものに限られます。

```
kde_padsynth/
├ sound_feature.py
└ sample/
  ├ sample1.wav
  ├ sample2.wav
  └ some_pack/
    ├ sound1.wav
    └ sound2.wav
```

以下のコマンドで合成を開始します。

```
python3 sound_feature.py
```

# パラメータ
- SoundFeature.__init__
  - interval
  - cutoff
  - spectrum_length
  - merging_range 未使用
- make_cdf
  - size
  - kernel_width
- padsynth
  - band_width
  - random_phase
- resynth
  - num_render
  - samplerate

# ライセンス
MIT
