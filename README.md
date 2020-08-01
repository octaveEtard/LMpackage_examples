# LMpackage_examples
Octave Etard ( octave.etard11@imperial.ac.uk )

This repo contains examples illustrating the use of the [LMpackage](https://github.com/octaveEtard/LMpackage) on real EEG datasets.

**Requires Matlab R2019b or newer** (tested on R2019b & R2020a).


## Installation
Add the `functions` folder(s) to your path. All function are prefixed with `LM_` so as to reduce the risk of shadowing any of your own functions. These examples require the [LMpackage](https://github.com/octaveEtard/LMpackage) to be in your path, as well as some [EEGLAB](https://sccn.ucsd.edu/eeglab/index.php) functions to load the data.


## Content

### `speech64_E`
This folder contains a 64-channel EEG dataset (in `data/EEG`) of 18 subjects listening to continuous speech. The stimulus was a 10-minute long, single-speaker audiobook, that was presented in four parts of roughly equal duration. The EEG data is divided in four parts for each subject, corresponding to each story part. The EEG was minimally pre-processed. It was simply average-referenced, and bandpass filtered between 1 and 12 Hz. The speech envelope (in `data/envelopes`) was computed by rectifying the speech waveform and lowpass filtering below 12 Hz. Both EEG and envelope were resampled to 100 Hz.

This dataset has previously been used in publications ([Etard, Kegler et al., NeuroImage 2019](https://www.sciencedirect.com/science/article/pii/S1053811919305208), [Etard & Reichenbach, J. Neurosci, 2019](https://www.jneurosci.org/content/39/29/5750)).

The examples in this folder demonstrates the use of [LMpackage](https://github.com/octaveEtard/LMpackage) to fit:
- a generic (population) forward model with leave-one-part-and-subject-out cross-validation (`forward_model.m`) using ridge-regularised regression. A forward model is trained using the data from all subjects bar one and all stimulus parts bar one. The model is then tested on the left out stimulus part for the left out subject. The left out subject and stimulus part are then iterated on until all testing combinations are exhausted. 

- subject-specific backward models with leave-one-part-out cross-validation for each subject (`backward_model.m`) using ridge-regularised regression.

(Note that in the case of this dataset a leave-one-part-out cross-validation is basically a four-fold cross-validation).
