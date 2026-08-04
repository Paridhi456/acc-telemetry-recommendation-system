import pandas as pd


df = pd.read_csv(
    "data/sector_data_with_time_loss_cleaned.csv",
    low_memory=False,
)

columns = [
    "BRAKE _Min",
    "BRAKE _Max",
    "BRAKE _Avg",
    "THROTTLE _Min",
    "THROTTLE _Max",
    "THROTTLE _Avg",
    "Corr Speed kmh_Min",
    "Corr Speed kmh_Max",
    "abs Steerangle_Min",
    "abs Steerangle_Max",
    "abs glat_Max",
    "abs glong_Max",
    "ROTY s_Min",
    "ROTY s_Max",
]

for column in columns:
    if column in df.columns:
        print("\n" + "=" * 60)
        print(column)
        print("=" * 60)

        print(
            df[column].describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.50,
                    0.95,
                    0.99,
                ]
            )
        )