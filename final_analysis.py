import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

fedspeak = pd.read_excel(open("fedspeak_data.xlsx","rb"))

fedspeak['Date'] = pd.to_datetime(fedspeak['Date'])
fedspeak['Date.1'] = pd.to_datetime(fedspeak['Date.1'])
fedspeak['Date.2'] = pd.to_datetime(fedspeak['Date.2'])

df1 = fedspeak[['Date', 'BENLPFED Index']].rename(columns={'Date': 'Date', 'BENLPFED Index': 'BENLPFED Index'})
df2 = fedspeak[['Date.1', 'SPX INDEX']].rename(columns={'Date.1': 'Date', 'SPX INDEX': 'SPX INDEX'})
df3 = fedspeak[['Date.2', 'LUATTRUU INDEX']].rename(columns={'Date.2': 'Date', 'LUATTRUU INDEX': 'LUATTRUU INDEX'})

combined_df = pd.concat([df1, df2, df3], ignore_index=True)
combined_df.dropna(how='all', inplace=True)
combined_df.sort_values(by='Date', inplace=True)
aggregated_df = combined_df.groupby('Date').agg({
    'BENLPFED Index': 'first',
    'SPX INDEX': 'first',
    'LUATTRUU INDEX': 'first'
}).reset_index()
aggregated_df.dropna(how='any',inplace=True)


negative = aggregated_df[(aggregated_df[["BENLPFED Index"]] < 0).all(axis=1)]

positive = aggregated_df[(aggregated_df[["BENLPFED Index"]] > 0).all(axis=1)]

def returns(k,df):
    series = df[k]
    change = series.pct_change()*100/(df['Date'].diff().dt.days)
    return change

sp_returns = returns("SPX INDEX",aggregated_df)
treasuries = returns("LUATTRUU INDEX",aggregated_df)
aggregated_df["SPX RETURNS"] = sp_returns
aggregated_df["LUATTRUU RETURNS"] = treasuries
aggregated_df.to_csv("modified_data.csv")
print(aggregated_df)

sp_returns = returns("SPX INDEX",negative)
treasuries = returns("LUATTRUU INDEX",negative)
aggregated_df["NEG SPX RETURNS"] = sp_returns
aggregated_df["NEG LUATTRUU RETURNS"] = treasuries



sp_returns = returns("SPX INDEX",positive)
treasuries = returns("LUATTRUU INDEX",positive)
aggregated_df["POS SPX RETURNS"] = sp_returns
aggregated_df["POS LUATTRUU RETURNS"] = treasuries

train, test = train_test_split(aggregated_df,test_size=0.5)
print(aggregated_df)
print("Positive S&P Returns:",train.loc[:,"POS SPX RETURNS"].mean())
print("Negative S&P Returns:",train.loc[:,"NEG SPX RETURNS"].mean())
print("Positive Treasury Returns:",train.loc[:,"POS LUATTRUU RETURNS"].mean())
print("Negative Treasury Returns:",train.loc[:,"NEG LUATTRUU RETURNS"].mean())