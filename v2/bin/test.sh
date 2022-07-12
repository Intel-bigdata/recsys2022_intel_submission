#bin/bash


python -u src/main.py \
    --dataset-dir ../datasets/$1 \
    --model-dir ../models_cp/$1 \
    --edge-drop 0.2 \
    --task $2 \
    --train-dataset $3 \
    --train-mode 'front_last' \
    --num-layers 2 --order 1 --embedding-dim 32 \
    --epochs 1 --num-workers 0 \
    --batch-size 3



# bash start.sh dressipi train 'train.txt'
# bash start.sh dressipi train 'train_small_23.txt,train_small_23.txt'
# bash start.sh dressipi eval 'train_small.txt'
# bash start.sh dressipi predict 'valid.txt'