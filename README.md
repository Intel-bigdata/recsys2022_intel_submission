## Introduction

Side Info Heterogeneous Graph for session recommender - SIHG4SR is a GNN-based solution for Recsys 2022 challenge.

## Contributions and Support

Since this project is for Recsys2022 submission, we will not accept external contributions and this software will not be supported going forward.

## LICENSE

This project is under MIT License

## Third Party Claim

The product includes content from several third-party sources that was originally governed by the licenses referenced below:

MSGIFSR - MIT License
https://github.com/SpaceLearner/SessionRec-pytorch/blob/main/LICENSE

## How to run

1. download dressipy dataset: https://www.dressipi-recsys2022.com/
2. use 'split_valid.py' to create a local valid dataset
3. follow README.md in v1, v2, v3, v3_enhanced for four leaderboard prediction
4. use ensemble.py to ensemble 4 predition cmdline
5. Now you have your final prediction - e2_2073.csv

## How to ensemble

```
# first level ensemble
python ensemble.py --task save_combine_df --path-list v3_2048.csv v2_2039.csv v1_2014.csv --weight-list 0.4 0.3 0.3 --save-path e1_2066.csv

# second level ensemble
python ensemble.py --task save_combine_df --path-list e1_2066.csv v3_enhanced_2064.csv --weight-list 0.6 0.4 --save-path e2_2073.csv
```

## Citation

```
@inproceedings{10.1145/3556702.3556852,
  author = {Xue, Chendi and Wang, Xinyao and Zhou, Yu and Ding, Ke and Zhang, Jian and Brugarolas Brufau, Rita and Anderson, Eric},
  title = {SIHG4SR: Side Information Heterogeneous Graph for Session Recommender},
  year = {2022},
  isbn = {9781450398565},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3556702.3556852},
  doi = {10.1145/3556702.3556852},
  abstract = {In this paper we present Side Information Heterogeneous Graph for Session Recommender – SIHG4SR, our solution for RecSys Challenge 2022[3], a competition organized by Dressipi for fashion recommendation. Dressipi provides data about user session, purchased items and content features to predict which fashion item will be bought. Our solution leverages side information and heterogeneous graph, deep dives into the data and engineers new features, employs two-stage training and multi-level ensemble strategy, and enhances the performance with fine tuning and hyper-parameter tuning. Finally SIHG4SR outperforms the state-of-art baselines, getting an MRR score 0.20762 and ranked 4th position on final leaderboard(team name ”MooreWins”). We published our solution at github1.},
  booktitle = {Proceedings of the Recommender Systems Challenge 2022},
  pages = {55–63},
  numpages = {9},
  keywords = {ACM RecSys Challenge 2022, Graph Neural Networks, Recommender System, End to End AI Optimization, Session-based Recommendation},
  location = {Seattle, WA, USA},
  series = {RecSysChallenge '22}
}
```
