import pandas as pd
import matplotlib.pyplot as plt


# Time of day
def categorize_time(time):
    if time.hour < 12:
        return "Morning"
    elif 12 <= time.hour < 17:
        return "Afternoon"
    elif 17 <= time.hour < 21:
        return "Evening"
    else:
        return "Night"

        # Season


# Load the data
df = pd.read_excel("ttc-subway-delay-data-2023.xlsx")

# Data cleaning
# Fill missing numeric values with the median
df["Min Delay"].fillna(df["Min Delay"].median(), inplace=True)
df["Min Gap"].fillna(df["Min Gap"].median(), inplace=True)

# For categorical data, you might fill missing values with the mode or a placeholder
df["Station"].fillna("Unknown", inplace=True)
df["Bound"].fillna("Unknown", inplace=True)

# Convert 'Date' to datetime and extract year, month, and day
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# Convert 'Time' to datetime and extract hour and minute
# Convert 'Time' to datetime using the 'HH:MM' format
df["Time"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce")
df["Time of Day"] = df["Time"].apply(categorize_time)


# Simple method using quantiles
upper_limit = df["Min Delay"].quantile(0.95)
df = df[df["Min Delay"] < upper_limit]

categorical_columns = ["Day", "Station", "Bound", "Time of Day"]


# One-hot encoding example
df = pd.get_dummies(df, columns=categorical_columns)


df.drop(["Date", "Time", "Code", "Min Gap", "Line", "Vehicle"], axis=1, inplace=True)


df.to_csv("cleaned_ttc_subway_data.csv", index=False)
