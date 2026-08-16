## Federated Anomaly Detection on IoT Traffic

This project compares a centralized anomaly detector against a federated one, using real botnet attack traffic captured from IoT devices. The goal was to understand how much (if any) detection accuracy you give up when you cannot centralize IoT traffic data across devices, which is the realistic constraint in most actual IoT deployments.

## Motivation

IoT devices generate huge amounts of traffic, and a lot of that traffic could be useful for training security models. But in practice, different devices often belong to different owners, companies, or networks, and pooling their raw traffic centrally raises privacy and bandwidth concerns. Federated learning offers a way around this: each device trains a local model on its own data, and only the model's learned parameters get shared with a central server, which aggregates them into an improved global model. Raw traffic never leaves the device.

This project builds and compares both approaches on the same problem, using the same underlying model, so the comparison is actually fair.

## Dataset

N-BaIoT, from the UCI Machine Learning Repository. It contains real network traffic captured from 9 commercial IoT devices, including both benign traffic and traffic recorded while the devices were infected with Mirai or BASHLITE botnet malware. Each row is a pre-engineered feature vector (115 statistical features per traffic window), so there is no raw packet parsing involved.

I used the Kaggle mirror of this dataset since it is easier to pull directly into Google Colab than the original UCI zip file.

I worked with 4 of the 9 devices to keep the project manageable within my timeline:

Danmini Doorbell
Ecobee Thermostat
Philips B120N10 Baby Monitor
SimpleHome XCS7 1002 Security Camera
What I actually did

Notebook 1, data exploration. Loaded benign and attack traffic for each device, checked for missing values (none), and confirmed that benign and attack traffic are statistically distinguishable by plotting feature distributions for both classes. Built a combined, labeled dataset across all 4 devices (1,344,470 rows total) and saved it for the next two notebooks to use.

Notebook 2, centralized baseline. Trained two anomaly detectors the standard way, with all data in one place. Both models were trained only on benign traffic and evaluated on a held out mix of benign and attack traffic, since in a real deployment you do not have labeled attack examples ahead of time.

Isolation Forest (via PyOD)
A small autoencoder (PyTorch), using reconstruction error as the anomaly score

Notebook 3, federated learning. Reframed the exact same problem using Flower. Each of the 4 devices was treated as a separate federated client, training the same autoencoder architecture locally on its own benign traffic for 5 epochs per round, over 10 rounds of FedAvg aggregation. No raw traffic was ever shared between clients or with the server, only model weights.

Results

Centralized:

Isolation Forest: 0.976 ROC-AUC
Autoencoder: 0.9999 ROC-AUC

Federated (FedAvg, 10 rounds, 4 devices):

Average across devices, final round: 0.99998 ROC-AUC
Per device, final round:
Danmini Doorbell: 0.99999
Ecobee Thermostat: 0.99999
Philips Baby Monitor: 0.99999
SimpleHome Security Camera: 0.99989

The federated version essentially matched the centralized autoencoder's performance. The gap is negligible, which suggests that for this dataset and this model, going federated does not meaningfully cost you detection accuracy, while still keeping raw traffic private to each device.

The security camera consistently scored slightly lower than the other three devices across every single round, though still very high overall. My best guess is that its normal traffic pattern is more variable than something like a thermostat's (which likely sends fairly repetitive traffic), making it a slightly harder baseline to model. I have not confirmed this, it is a hypothesis based on the pattern in the results.

The federated model also converged fast. The average ROC-AUC barely changed after round 1 and stayed roughly flat through round 10. This is a good property for a real deployment since it implies you would not need many communication rounds between devices and the server to reach strong performance.

Honest limitations

I want to be upfront about a few things rather than just presenting the headline numbers:

These results are consistent with what published papers on N-BaIoT report. The dataset's attack traffic is statistically very distinct from benign traffic, which is part of why both models score this high. This says as much about the dataset as it does about the models.
Benign class recall was 90 percent (Isolation Forest) and 95 percent (autoencoder), not 100 percent, even though ROC-AUC looks close to perfect. A high AUC can hide the fact that a meaningful fraction of normal traffic is still being misclassified.
I evaluated on a subset of attack variants (2 out of 5 available per malware family, gafgyt and mirai) for each device, not the full range of attacks available in the dataset. The models have not been tested against attack variants they have never seen at all. A natural next step would be testing generalization to held out attack types specifically, rather than just held out traffic samples of known attack types.
The federated setup here is a simulation running on one machine (via Flower's simulation mode with Ray), not an actual multi device deployment. It demonstrates the algorithmic approach correctly, but does not capture real world factors like network latency or devices dropping out mid round.
How to run this

All 3 notebooks are built for Google Colab.

Get a Kaggle API token: kaggle.com, then Settings, then API Tokens, then Generate New Token.
In Colab, click the key icon in the left sidebar, add a secret named KAGGLE_TOKEN with your token as the value, and turn on notebook access.
Run the notebooks in order. Each one saves intermediate output (to Google Drive, under gri-project1/processed) that the next notebook loads, so they need to be run in sequence, and Drive needs to stay mounted across sessions if you want to avoid re downloading the dataset each time.
What I would extend this toward

Given more time, I would want to test the models against attack variants not seen during evaluation at all, try a non IID federated split (right now each device's local data is just its own natural traffic, but a harder and more realistic test would deliberately skew the label distribution per client), and compare FedAvg against a more robust aggregation strategy under simulated data poisoning, since that is closer to the kind of question real federated deployments have to answer.
