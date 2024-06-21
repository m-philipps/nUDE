import pandas as pd
import petab

training_df = petab.get_measurement_df("training_dataset.tsv")
validation_df = petab.get_measurement_df("validation_dataset.tsv")

df = pd.concat([
    training_df.loc[training_df.time <= 20],
    validation_df.loc[validation_df.time > 20],
])

petab.write_measurement_df(df=df, filename="dataset.tsv")
