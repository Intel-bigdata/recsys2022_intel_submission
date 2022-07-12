### Prepare develope enviroment
```
conda env create -f conf/environment_gpu.yml
```


### Data preprocess
1. Download the challenge data into the `datas` folder under current folder: https://www.dressipi-recsys2022.com/
2. Use 'split_valid.py' in parent folder to create a local valid dataset， and move it to the `datasets/dressipi` folder and `datas` folder under current folder
3. Run the following command to produce preprocessed data into the folder `datasets/dressipi` before training
```
# before local validation
sh bin/preprocess_local.sh

# before leaderboard validation
sh bin/preprocess_leaderboard.sh
```

### Training and evaluation
- for local validation
    - run the following command, it will produce one prediction file in `$model_dir`, named `sub_pt_ft.csv`, it will be used for ensemble evaluation. All the training and validating metric will be stored in a file named `run.log` under the `$model_dir`.
```
sh bin/pipline_local.sh
```

- for leaderboard validation
    - run the following command, it will produce two prediction file in `$model_dir`, named `sub_ldb.csv` and `sub_final.csv`, for test leaderboard and final leaderboard respectively.
```
sh bin/pipline_lederboard.sh
```
