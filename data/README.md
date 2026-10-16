# Data

The app expects the Kaggle-style *Airline Passenger Satisfaction* training file here, as either

- `data/train.csv`, or
- `data/train.zip` (a zip containing `train.csv`; pandas reads it directly).

Expected columns (23): `id`, `Gender`, `Customer Type`, `Age`, `Type of Travel`, `Class`, `Flight Distance`, the 13 service ratings,
`Departure Delay in Minutes`, `Arrival Delay in Minutes`, and the target `satisfaction` (True/False).

The dataset is used by the **Analytics & Insights** page and by `train_models.py`. The Prediction Center and Project Overview
only need the files in `models/`.

Do not commit confidential or restricted data. Check the dataset's licence before publishing `train.zip`; if redistribution is not
allowed, delete it, keep the `data/*.csv` rule in `.gitignore`, and describe where to download it instead.
`test.csv` from the notebook is not needed: it has no labels and is not part of the evaluation here.
