data_dir='/path/to/your/data'

## stage1 train
save_path='/path/to/save/model'
python -u src/scripts/main.py \
        --save_path  $save_path \
        --dataset-dir $data_dir \
        --train-session train_sessions.csv \
        --train-purchase train_purchases.csv 

# stage2 train
save_path='/path/to/save/model'
model_path='/model/generated/from/stage1'
python -u src/scripts/main.py \
        --save_path  $save_path \
        --epochs 1 \
        --resume-train \
        --model $model_path \
        --dataset-dir $data_dir \
        --use-recent-n-month 0.73 \
        --lr 0.0005539757616384248 \
        --weight-decay 0.000001 \
        --gamma 0.3195666887614495 \
        --step_size 2 \
        --train-session train_sessions.csv \
        --train-purchase train_purchases.csv 

# predict test
save_path='/path/to/save/prediction'
model_path='/model/generated/from/stage2'
python -u src/scripts/main.py \
        --save_path  $save_path \
        --model $model_path \
        --dataset-dir $data_dir \
        --predict \
        --valid-session 'test_leaderboard_sessions.csv' \
        --valid-purchase ""  \
        --candidate-list 'candidate_items.csv'

# ## predict final
save_path='/path/to/save/prediction'
model_path='/model/generated/from/stage2'
python -u src/scripts/main.py \
        --save_path  $save_path \
        --model $model_path \
        --dataset-dir $data_dir \
        --predict \
        --valid-session 'test_final_sessions.csv' \
        --valid-purchase ""  \
        --candidate-list 'candidate_items.csv'