## Introduction

Side Info Heterogeneous Graph for session recommender - SIHG4SR is a GNN-based solution for Recsys 2022 challenge.

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
