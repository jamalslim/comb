# data

The data are not stored in the repository; the scripts write them here.

* Showers: `python scripts/comb_ds1.py prepare` downloads CaloChallenge Dataset 1 (photons, Zenodo record 8099322,
  about 0.5 GB including the raw file `raw/dataset_1_photons_1.hdf5`) and writes the prepared splits here. The raw file
  (174 MB) is larger than GitHub accepts.
* Digits: `python scripts/comb_strokes.py prepare --data uci` turns the UCI digits, which ship with scikit-learn,
  into pen strokes (`strokes_uci.npz`) and trains the digit classifier used to judge samples.
