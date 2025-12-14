import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_csv("cs-training.csv", index_col=0)
print("Loaded:", df.shape)

# Target: 1 = default, 0 = non-default
# App cần: 1 = GOOD, 0 = BAD
df['Target'] = 1 - df['SeriousDlqin2yrs']

# ==========================================
# 2. CLEAN DATA
# ==========================================
df = df[df['age'] > 18]

# Fill missing income bằng median theo tuổi
df['MonthlyIncome'] = df.groupby('age')['MonthlyIncome']\
    .transform(lambda x: x.fillna(x.median()))

df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)

# Cap outliers
df['DebtRatio'] = df['DebtRatio'].clip(0, 5)
df['MonthlyIncome'] = df['MonthlyIncome'].clip(0, 50_000)

# ==========================================
# 3. MAP SANG FEATURE APP
# ==========================================

# Age
df['Age'] = df['age']

# Credit amount (proxy tổng nghĩa vụ tài chính)
df['Credit amount'] = (
    df['MonthlyIncome'] * (1 + df['DebtRatio'])
).clip(500, 20_000)

# Duration (giả lập thời hạn vay)
np.random.seed(42)
df['Duration'] = np.random.choice(
    [12, 24, 36, 48, 60],
    size=len(df),
    p=[0.15, 0.25, 0.25, 0.2, 0.15]
)

# Telco Bill (proxy hành vi chi tiêu)
df['Telco_Bill'] = (
    df['MonthlyIncome'] * np.random.uniform(0.05, 0.12, len(df))
).clip(50_000, 2_000_000)

# Social Score (KHÔNG dùng Target)
df['Social_Score'] = (
    100
    - df['DebtRatio'] * 20
    - df['NumberOfTimes90DaysLate'] * 10
    - df['NumberOfTime60-89DaysPastDueNotWorse'] * 7
    + np.random.normal(0, 8, len(df))
).clip(10, 95)

# ==========================================
# 4. FINAL DATASET
# ==========================================
final_df = df[
    ['Age', 'Credit amount', 'Duration', 'Telco_Bill', 'Social_Score', 'Target']
]

final_df.to_csv("final_thesis_data.csv", index=False)

print("✅ Preprocessing hoàn tất")
print(final_df.head())
print(final_df['Target'].value_counts(normalize=True))
