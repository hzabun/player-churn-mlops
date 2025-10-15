import pandas as pd

trainLabelCSVPath = "data/label/train_labeld.csv"
df = pd.read_csv(trainLabelCSVPath)

print(df.shape[0])
