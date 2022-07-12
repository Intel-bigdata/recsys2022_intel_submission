# SIHG4SR V1
This repo contains the implementation for SIHG4SR.

# Requirements
The performance is tested on following environments:
- dgl 0.6.1
- pytorch 1.7.1
- pandas 1.4.2

# Usage
First please download the dataset from [RecSys Challenge 2022](https://www.recsyschallenge.com/2022/).
Then you can refer to [run.sh](run.sh), or follow the steps as below. 

## Stage1 training
In first stage, we train with all the training dataset:
```
python scripts/main.py --save_path '/path/to/save/model' --dataset-dir '/path/to/your/data' --train-session train_sessions.csv  --train-purchase train_purchases.csv 
```

## Stage2 training
In second stage, we train only with last month of training dataset and change the optimizer settings:
```
python -u scripts/main.py --save_path '/path/to/save/model' --epochs 1 --resume-train --model '/model/generated/from/stage1' --dataset-dir '/path/to/your/data' --use-recent-n-month 0.73 --lr 0.0005539757616384248 --weight-decay 0.000001 --gamma 0.3195666887614495 --step_size 2 --train-session train_sessions.csv --train-purchase train_purchases.csv 
```

## Prediction
Then we can use the trained model to predict the test or final data.
```
# predict test
python -u scripts/main.py  --save_path '/path/to/save/prediction'  --model '/model/generated/from/stage2'  --dataset-dir '/path/to/your/data'  --predict  --valid-session 'test_leaderboard_sessions.csv'  --valid-purchase ""   --candidate-list 'candidate_items.csv'

# predict final
python -u scripts/main.py  --save_path  '/path/to/save/prediction'  --model '/model/generated/from/stage2'  --dataset-dir '/path/to/your/data'  --predict  --valid-session 'test_final_sessions.csv'  --valid-purchase ""   --candidate-list 'candidate_items.csv'
```
The predicted csv file will be saved to your save_path.

## Local performance test
If you want to do a local performance test, please first split the training dataset accorrding to [split_valid.py](../split_valid.py), then refer [run_localtest.sh](run_localtest.sh)

