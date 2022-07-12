import pandas as pd

filenames = ["train_purchases",  "train_sessions"]
root_path = "dressipi/"
data = dict((n, pd.read_csv(f"{root_path}/{n}.csv")) for n in filenames)

valid_sessions_new = data['train_sessions'][data['train_sessions']['date'] > '2021-05-01']
valid_purchases_new = data['train_purchases'][data['train_purchases']['date'] > '2021-05-01']
valid_candidates = valid_purchases_new.drop_duplicate

valid_sessions_new.to_csv(f"{root_path}/valid_sessions_new.csv", index=False)
valid_purchases_new.to_csv(f"{root_path}/valid_purchases_new.csv", index=False)
valid_candidates.to_csv(f"{root_path}/valid_candidate_items.csv", index=False)
