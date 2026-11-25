# ECG Biometrics Across Recording Sessions

*Signal Analysis and Representations — course project report*

Can a single ECG channel identify a person or verify a claimed identity when the enrollment and query recordings come from different sessions? We compared hand-crafted signal features with a small 1D convolutional neural network (CNN) and tested how both representations respond to degraded recordings. On two larger held-out cohorts, CNN embeddings outperformed PCA for people the model had never seen. Noise augmentation improved robustness, but verification errors remained too high for reliable authentication.

## Data and methods

We used [ECG-ID v1.0.0](https://physionet.org/content/ecgiddb/1.0.0/) (309 prepared recordings), [Heartprint v3](https://doi.org/10.6084/m9.figshare.20105354.v3) (1,400), and the repeated-record subset of [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) (5,041). Each example was a whole original recording. We removed 139 exact waveform duplicates from Heartprint before splitting. The preprocessing pipeline resampled each single-channel recording to 250 Hz, applied a 0.5–40 Hz band-pass filter, centered the signal, and normalized its RMS amplitude.

The classical model used power spectra, autocorrelation, amplitude statistics, and estimated RR intervals, standardized and reduced to 16 PCA components. The 1D-CNN was trained on participant IDs for 20 epochs and produced a 64-dimensional embedding per recording. Both approaches compared queries to participant templates with cosine similarity. We also tested normalized cross-correlation and dynamic time warping (DTW) as waveform baselines.

In the **known-participant** protocol, the first session supplied training and enrollment examples; later sessions supplied test queries. In the **unseen-participant** protocol, the training, validation, and test participants were disjoint. The first session of each held-out participant supplied enrollment templates without updating the model; later sessions supplied queries. Identification ranked the full gallery. Verification scored one genuine claim and up to ten randomly sampled impostor claims per query. We report equal error rate (EER) and, separately, false acceptance and rejection at an operating threshold set on validation.

## Results

The table reports test performance before augmentation. Top-1 is the proportion of queries assigned to the correct participant. Lower EER is better.

| Dataset and protocol | Gallery / queries | PCA top-1 | CNN top-1 | PCA EER | CNN EER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heartprint, known | 197 / 924 | **16.2%** | 6.6% | **22.3%** | 29.5% |
| Heartprint, unseen | 39 / 176 | 40.3% | **58.5%** | 22.2% | **13.4%** |
| PTB-XL, known | 2,042 / 2,824 | **15.0%** | 5.7% | **20.0%** | 24.8% |
| PTB-XL, unseen | 408 / 577 | 23.7% | **32.2%** | 19.4% | **13.2%** |

On the same ten-person unseen Heartprint subset, CNN reached 84.2% top-1, PCA 44.7%, DTW 28.9%, and cross-correlation 13.2%. These figures apply only to that smaller gallery. The unseen ECG-ID test contained four participants and was treated as a pipeline check. The known and unseen scores cannot isolate the effect of training on the same person: gallery size and the number of available training examples per person also change.

## Robustness and limitations

We added synthetic 0.25 Hz drift or band-limited Gaussian noise to held-out queries **after** preprocessing. Enrollment signals and validation thresholds stayed fixed across test conditions for each model. The augmented CNN saw a mixture of original and perturbed training recordings. With band-limited noise at 50% of the original signal RMS (about 6 dB SNR), augmentation raised top-1 from **36.4% to 50.6%** on Heartprint and from **9.9% to 20.6%** on PTB-XL. On the latter, at each model's validation threshold, false rejection fell from 31.2% to 15.3%, while false acceptance rose from 13.0% to 13.9%.

The results show that single-channel ECG contains useful identity information across sessions, and that augmentation can help retain it in noisy queries. The study used one split and training seed. It did not test transfer across sensors or sites, and the simulated noise does not reproduce all acquisition artifacts. The most useful follow-up would repeat splits and initializations and test with recordings collected on a different device.

See the [reproduction instructions](README.md), [experimental protocol](PROTOCOL.md), [full baseline results](RESULTS.md), and [robustness results](ROBUSTNESS.md). The repository contains code and aggregate results only; obtain the original datasets from their publishers.
