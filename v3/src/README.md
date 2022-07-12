## prepare data
### prepare valid dataset
python ../../split_valid.py

### prepare item_feature_extra.csv and categorical_item_features.csv
sh prepare_extra_csv.sh

## Run training
sh test_full.sh
