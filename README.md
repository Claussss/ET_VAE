# Audio Effect Transfer Using VAEs

## Overview
This project presents a machine learning system aimed at transferring audio effects from one monophonic audio source to another. Nowadays, applying effects to sound is a ubiquitous practice, enhancing the pleasantness or serving as a contrast to improve other sounds. This system is particularly valuable in the music industry, where sounds in songs often undergo the application of multiple effects.

You can find some model showcases in the playground.ipynb.

## Key Features
- **Automated Audio Effects Transfer**: The system efficiently replicates audio effects from one sound to another, overcoming the laborious process involved in manual replication.
- **Innovative Use of VAE**: The backbone of the system is a Variational Autoencoder (VAE), which generates a latent space specifically tailored for manipulating audio effects.
- **Unique Loss Term**: Our system includes a novel loss term that successfully separates the timbre, pitch, and various audio characteristics of the original sound from the effects applied to it.

## Installation
Clone the repository and install the necessary dependencies.

```bash
git clone [repository-link]
cd [repository-name]
pip install -r requirements.txt
```

## Dataset Generation
A python script is provided to generate a dataset.

### Script Usage
The script `generate_stft_specs.py` in the `./scripts` directory can be used to create datasets with different audio transformations. It accepts the following parameters:

```bash
python ./scripts/generate_stft_specs.py [dataset-type] [spectrogram-type]
```

- `dataset-type`: Choose from `test` or `valid` to specify the type of dataset you wish to generate.
- `spectrogram-type`: Specify the type of spectrogram. Options include `stft` (short-time Fourier transform), `cqt` (constant-Q transform), `mel` (Mel-scaled spectrogram), or `hifi` (high-fidelity).

### Example
To generate a test dataset with Mel-scaled spectrograms:

```bash
python ./scripts/generate_stft_specs.py test mel
```
