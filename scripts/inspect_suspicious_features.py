import pandas as pd


df = pd.read_csv(
    "data/sector_data_with_time_loss_cleaned.csv",
    low_memory=False,
)

columns = [
    "PID",
    "Lap",
    "Sector",
    "THROTTLE _Min",
    "THROTTLE _Max",
    "THROTTLE _Avg",
    "abs Steerangle_Min",
    "abs Steerangle_Max",
    "abs glat_Max",
    "abs glong_Max",
    "ROTY s_Min",
    "ROTY s_Max",
]

print("\nRows where throttle exceeds 100:")
print(
    df.loc[
        df["THROTTLE _Max"] > 100,
        columns,
    ]
    .head(30)
    .to_string(index=False)
)

print("\nRows where abs steering is negative:")
print(
    df.loc[
        (
            (df["abs Steerangle_Min"] < 0)
            | (df["abs Steerangle_Max"] < 0)
        ),
        columns,
    ]
    .head(30)
    .to_string(index=False)
)

print("\nRows where abs longitudinal G is negative:")
print(
    df.loc[
        df["abs glong_Max"] < 0,
        columns,
    ]
    .head(30)
    .to_string(index=False)
)

print("\nRows with extreme ROTY:")
print(
    df.loc[
        (
            df["ROTY s_Min"].abs() > 100
        )
        | (
            df["ROTY s_Max"].abs() > 100
        ),
        columns,
    ]
    .head(30)
    .to_string(index=False)
)