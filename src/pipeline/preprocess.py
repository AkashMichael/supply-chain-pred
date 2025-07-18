import pandas as pd

# Load data
df = pd.read_excel("notebook/data/Retail-Supply-Chain-Sales-Dataset.xlsx")

# Convert Order Date
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Aggregate demand per day (using 'Sales' or 'Quantity')
daily_df = df.groupby("Order Date")["Quantity"].sum().reset_index()
daily_df.rename(columns={"Quantity": "demand"}, inplace=True)

# Create lag features
daily_df["lag_1"] = daily_df["demand"].shift(1)
daily_df["lag_7"] = daily_df["demand"].shift(7)
daily_df["lag_30"] = daily_df["demand"].shift(30)

# Rolling mean
daily_df["rolling_mean_7"] = daily_df["demand"].rolling(window=7).mean()

# Extract month and day of week
daily_df["month"] = daily_df["Order Date"].dt.month
daily_df["day_of_week"] = daily_df["Order Date"].dt.dayofweek

# Drop NaN rows due to shifting/rolling
daily_df.dropna(inplace=True)

# Save as preprocessed file (overwrite or create new)
daily_df.to_excel("notebook/data/preprocessed_dataset.xlsx", index=False)
