# Federated Anomaly Detection on IoT Traffic

This project compares a centralized anomaly detector against a federated one, using real botnet attack traffic captured from IoT devices. The goal was to understand how much (if any) detection accuracy you give up when you cannot centralize IoT traffic data across devices, which is the realistic constraint in most actual IoT deployments.

**Summary of Findings:** on the first, narrow evaluation, federated learning looked like it cost nothing (0.99998 AUC). Extending to all 10 attack types, that still holds for 8 of them: every method, including plain FedAvg, scores 0.9998 to 1.0000. The weaker pooled number for plain FedAvg (0.95 AUC) comes almost entirely from two near-duplicate attack files (`gafgyt_tcp`, `gafgyt_udp`, about 20 distinct rows each), where plain FedAvg scores 0.66 to 0.75 and FedProx or local scaling score 0.97 to 0.98. Details and caveats below.

## Motivation

IoT devices generate huge amounts of traffic, and a lot of that traffic could be useful for training security models. But in practice, different devices often belong to different owners, companies, or networks, and pooling their raw traffic centrally raises privacy and bandwidth concerns. Federated learning offers a way around this: each device trains a local model on its own data, and only the model's learned parameters get shared with a central server, which aggregates them into an improved global model. Raw traffic never leaves the device.

This project builds and compares both approaches on the same problem, using the same underlying model, so the comparison is actually fair.

## Dataset

[N-BaIoT](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot), from the UCI Machine Learning Repository. It contains real network traffic captured from 9 commercial IoT devices, including both benign traffic and traffic recorded while the devices were infected with Mirai or BASHLITE (gafgyt) botnet malware. Each row is a pre-engineered feature vector (115 statistical features per traffic window), so there is no raw packet parsing involved.

I used the [Kaggle mirror](https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset) of this dataset since it is easier to pull directly into Google Colab than the original UCI zip file.

I worked with 4 of the 9 devices:
- Danmini Doorbell
- Ecobee Thermostat
- Philips B120N10 Baby Monitor
- SimpleHome XCS7 1002 Security Camera

## Repo contents

| File | What it is |
|---|---|
| `01_data_exploration.ipynb` | Notebook 1: data exploration, builds the labeled dataset |
| `02_baseline_centralized.ipynb` | Notebook 2: centralized baseline (Isolation Forest and autoencoder) |
| `03_federated_flower.ipynb` | Notebook 3: federated learning with Flower |
| `04_extension_noniid_unseen.ipynb` | Extension: all 10 attack types, non-IID experiments, unseen devices |

## What I did

### Notebooks 1 to 3: first pass

**Notebook 1, data exploration.** Loaded benign and attack traffic for each device, checked for missing values (none), and plotted feature distributions for both classes. Built a combined, labeled dataset across all 4 devices (1,344,470 rows) and saved it for the next parts. This dataset used 4 attack types per device (2 gafgyt, 2 mirai: `gafgyt_combo`, `gafgyt_junk`, `mirai_ack`, `mirai_scan`).

![Benign vs attack feature distributions](images/feature_distributions.png)

**Notebook 2, centralized baseline.** Trained two anomaly detectors with all data in one place. Both were trained only on benign traffic and evaluated on held out benign plus attack traffic, since in a real deployment you do not have labeled attack examples ahead of time.
- Isolation Forest (via PyOD)
- A small autoencoder (PyTorch), using reconstruction error as the anomaly score

**Notebook 3, federated learning.** Reframed the same problem using Flower. Each of the 4 devices was a separate federated client training the same autoencoder on its own benign traffic, with FedAvg aggregation over 10 rounds. Each client ran 5 full-batch gradient steps per round (not 5 passes over minibatches). No raw traffic was shared, only model weights.

### Extension: harder questions

The first results looked almost perfect, so I went back and stress tested them. The extension notebook asks three questions:

1. **Attack coverage.** Does the model still work on all 10 attack types per device, not just the 4 used in the first evaluation?
2. **Non-IID data.** The devices differ a lot in traffic patterns and in data size (thermostat about 9k benign training rows, baby monitor about 123k). Does plain FedAvg suffer, and do FedProx, equal client weighting, or local scaling help?
3. **Unseen devices.** If a new device joins after training, does the global model work on it?

Notes on the setup:
- Every model trains on benign traffic only, so *every* attack is unseen at training time. The first evaluation's gap was coverage (6 of 10 attack types were never tested), not training exposure.
- Because training is benign only, labels cannot be skewed across clients. The non-IID skew here comes from the devices themselves: different feature scales, traffic patterns and data sizes.
- Federated training in this notebook is a plain PyTorch loop instead of Flower simulation, so many configurations can be run quickly. It implements the same FedAvg algorithm.
- Local training uses real minibatches (batch size 256), Adam, lr 1e-3, 10 rounds.
- Every result is the average of 3 seeds, shown with the standard deviation across seeds.
- Each attack file is capped at 20,000 random rows to fit in Colab memory.
- The shared feature scaler is built from per-client row counts, sums and sums of squares, so no raw rows are shared. It matches a scaler fit on pooled data.
- The detection threshold is the 95th percentile of each device's own benign training error.

## Results

### First pass (4 attack types)

**Centralized:**
- Isolation Forest: 0.976 ROC-AUC
- Autoencoder: 0.9999 ROC-AUC

**Federated (FedAvg, 10 rounds, 4 devices):**
- Average across devices, final round: 0.99998 ROC-AUC
- Per device: Doorbell 0.99999, Thermostat 0.99999, Baby Monitor 0.99999, Security Camera 0.99989

On this evaluation, federated matched centralized. The extension below shows this holds for 8 of the 10 attack types, but not for the two remaining ones (`gafgyt_tcp`, `gafgyt_udp`), which this first evaluation never tested.

### Extension: all 10 attack types

Mean ROC-AUC over the 4 devices, all 10 attack types pooled, mean and standard deviation over 3 seeds:

| Method | AUC | Attack recall (TPR) |
|---|---|---|
| FedProx (mu=0.1), 5 local epochs | 0.9959 +- 0.0013 | 0.966 |
| FedAvg, local scalers | 0.9935 +- 0.0004 | 0.950 |
| Local only (no federation) | 0.9910 +- 0.0025 | 0.950 |
| Centralized (pooled) | 0.9877 +- 0.0022 | 0.883 |
| **FedAvg, 1 local epoch, shared scaler** | **0.9507 +- 0.0246** | 0.883 |
| FedAvg, equal client weights | 0.9420 +- 0.0118 | 0.850 |
| FedAvg, 5 local epochs | 0.9326 +- 0.0050 | 0.883 |

Per device AUC (mean over seeds):

| Method | Doorbell | Thermostat | Baby Monitor | Camera |
|---|---|---|---|---|
| FedAvg, 1 local epoch | 0.9979 | 0.8974 | 0.9906 | 0.9167 |
| FedAvg, local scalers | 0.9981 | 0.9968 | 0.9974 | 0.9817 |
| FedProx, 5 local epochs | 0.9991 | 0.9994 | 0.9971 | 0.9881 |

What this shows:

- **Two attack types break plain FedAvg.** For the main FedAvg model, 8 of the 10 attack types score about 1.00 on every device. `gafgyt_tcp` and `gafgyt_udp` score 0.99 (Doorbell), 0.95 (Baby Monitor), 0.58 (Camera) and 0.49 (Thermostat). An AUC near 0.5 means no better than a coin flip on that attack. These two attack files are near-copies with only about 20 distinct rows each, one weak signal that makes up 20% of the pooled attack rows. Plain FedAvg fits normal Thermostat and Camera traffic poorly enough that benign error rises above this attack's error, which is why AUC drops below 0.5. Local-only models rank these attacks better (seed 0: 0.86 Thermostat, 0.92 Camera, 0.99 Doorbell and Baby Monitor), but at the 95th-percentile threshold they still catch almost none on the Thermostat and Camera. So these two attacks are hard for every method on those devices, not only for plain FedAvg.
![Thermostat reconstruction-error histograms](images/thermostat_error_profile.png)
- **Plain FedAvg is the weakest method, and also the least stable** (standard deviation 0.025 across seeds, versus 0.0004 to 0.0025 for the better methods).
- **Handling device differences fixes most of it.** Giving each device its own feature scaler, or using FedProx, brings the average back to 0.994 to 0.996. Thermostat AUC goes from 0.897 to 0.997 or higher.
- **The better methods are close to each other.** FedProx, local scalers and local-only differ by about as much as the seed-to-seed spread, so I do not claim one is better than the others. The main finding is that plain FedAvg with a shared scaler is clearly worse than all of them.
- **Equal client weighting did not help**, so data size imbalance alone does not seem to be the cause. My guess is that differences in feature scale between devices matter more (local scaling fixes it), but I have not confirmed this.
- **Pooled training is not an upper bound here.** It scored below local-only models, probably because one small autoencoder has to model four very different devices at once.

### Extension: unseen devices

Leave one device out: train FedAvg on 3 devices, then test on the fourth. AUC on the new device:

| Held-out device | Reuse fleet scaler | Compute its own scaler |
|---|---|---|
| Doorbell | 0.9974 | 0.9965 |
| Thermostat | 0.8616 | 0.9855 |
| Baby Monitor | 0.9230 | 0.9820 |
| Camera | 0.8782 | 0.9814 |

A new device works well if it computes its own normalization from its first benign traffic. Reusing the other devices' normalization does not.

### Note on benign recall

The detection threshold is the 95th percentile of benign training error, so benign recall (TNR) is about 0.95 for every method by construction. It is a result of the threshold choice, not a model weakness. Attack recall (TPR) at that threshold is the more informative number, and it varies from 0.85 to 0.97 between methods.

## Model and setup details

Autoencoder architecture: input (115 features) -> 64 -> 16 (latent) -> 64 -> 115, ReLU activations, MSE loss, Adam (lr=1e-3).

Isolation Forest: contamination=0.1, PyOD's default implementation.

Centralized baseline (Notebook 2): 30 epochs, batch size 256, 70/30 split on benign data only (random, seed 42), with all attack samples added to the test set.

Federated, Notebook 3 (Flower): FedAvg, 10 rounds, 5 full-batch gradient steps per client per round, all 4 clients every round, 70/30 benign split in file order.

Federated, extension: see "Notes on the setup" above. The 70/30 benign split is in file order, as in Notebook 3.

## Honest limitations

- **Single run per setting, 3 seeds.** The ranking of the top three methods is not settled. Only the gap between them and plain FedAvg is clearly larger than the noise.
- **The cause of the FedAvg failure is a hypothesis.** Local scaling and FedProx both fix it, which points to differences between devices, but I have not run a test that isolates the exact cause.
- **The two hard attacks are one weak, repeated signal.** `gafgyt_tcp` and `gafgyt_udp` have only 17 to 27 distinct rows per 20,000 and are near-copies of each other. Under a shared model their reconstruction errors are identical on all four devices, which suggests the same fixed point on every device (not directly verified). They make up 20% of the pooled attack rows, so they weigh heavily in pooled AUC and TPR despite carrying little information. Single raw features separate them from benign traffic almost perfectly, so the failure is not that they resemble normal traffic. It comes from the global model fitting Thermostat and Camera benign traffic poorly, which lifts normal error above this weak attack's error. Results on these two attacks also swing with the seed.
- Method rankings on `gafgyt_tcp`/`gafgyt_udp` depend on how each model happens to reconstruct essentially one point (its error ranges from 0.07 to 4.4 across methods), so they should not be read as a general detection-quality ranking.
- **The dataset is easy.** Attack traffic is statistically very distinct from benign traffic for most attack types, which is why most scores are near 1.0. Published N-BaIoT papers report similarly high numbers.
- **Attack files are subsampled** to 20,000 rows each in the extension. AUC should be stable at that size, but I did not test other sizes.
- **The federated setup is a simulation on one machine**, not a real multi-device deployment. It does not capture latency, dropped devices, or communication cost.
- **Only 4 of the 9 devices** were used.
- **Notebook 3's "5 epochs" is 5 full-batch gradient steps,** far less training than the centralized model got. The extension fixes this with minibatch training.

## How to run this

All notebooks are built for Google Colab.

1. Get a Kaggle API token: kaggle.com, then Settings, then API Tokens, then Generate New Token.
2. In Colab, click the key icon in the left sidebar, add a secret named `KAGGLE_TOKEN` with your token as the value, and turn on notebook access.
3. Run notebooks 01, 02 and 03 in order. Each one saves intermediate output to Google Drive under `nbaiot-project/processed` for the next one to load.
4. Run `04_extension_noniid_unseen.ipynb`. It downloads the raw data itself (about 1.75 GB, skipped if already present) and does not need steps 1 to 3. With the default 3 seeds it took roughly 15 to 20 minutes on Colab CPU. It saves four CSV result files to the same Drive folder.

## What I would extend this toward

- Run a test that isolates why plain FedAvg fails on `gafgyt_tcp` and `gafgyt_udp` (for example, by varying feature scaling while holding everything else fixed), and check why reconstruction error misses attacks that single features separate perfectly.
- Compare FedAvg against a more robust aggregation strategy (median or trimmed mean) under simulated data poisoning, where one client sends bad updates. This is closer to the kind of question real federated deployments have to answer.
- Use all 9 devices and a threshold chosen without any labeled data from other devices.