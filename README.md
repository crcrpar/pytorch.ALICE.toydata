**OFFICIAL REPOSITORY IS [HERE](https://github.com/ChunyuanLI/ALICE).**  
**This is not official and does not reproduce/implement all the experiments and results.**  

# ALICE
Adversarially Learned Inference with Conditional Entropy.

This repository has 2 kinds of experiments on GMM dataset.

## Requirements
- Python 3.6.5
- matplotlib
- pytorch 0.4.1
- tqdm (progress bar)

## How to use
1. `python3 train_ALICE_toydata.py`: Train using *explicit cyclic consistency*
2. `python3 train_ALICE_toydata.py --adv`: Train using *implicit cyclic consistency*

`--easy` option reduces the number of modes in dataset `X`.

Results and the used dataset saved under `args.results_dir/{YYMMDD-HMS}ALICE_unsupervised_{MSE or adversarial}_reconstruction`

Datasets are saved [x, z]\_trn.pkl using torch.save method.

- Default dataset:  
![figure1](https://github.com/crcrpar/pytorch.ALICE.toydata/blob/master/results/180909-195705ALICE_unsupervised_MSE_reconstruction/dataset.png?raw=true)
- `--easy`:  
![figure2](https://user-images.githubusercontent.com/16191443/45264048-e9675400-b470-11e8-9b36-5b4b6fa32ba6.png)

## Results
Some results are under `results`.
### Explicit Cyclic Consistency
![figure3](https://github.com/crcrpar/pytorch.ALICE.toydata/blob/master/results/180909-195705ALICE_unsupervised_MSE_reconstruction/reconstructed_100.png?raw=true)

### Implicit Cyclic Consistency
I emprically confirmed `--n_dis` had some effects on results though, training in this setting was not stable in general.
![figure4](https://github.com/crcrpar/pytorch.ALICE.toydata/blob/master/results/180909-195933ALICE_unsupervised_adversarial_reconstruction/reconstructed_100.png?raw=true)

## What I implement
- Experiments on toy-dataset (GMM)
