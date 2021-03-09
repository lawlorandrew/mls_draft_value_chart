import pandas as pd

years = range(2016,2021)

stats_df = pd.DataFrame()
for year in years:
  df = pd.read_csv(f'data/season-stats/{year}.csv')
  df = df.groupby('Player')[['Min']].sum()
  df = df.reset_index()
  df['year'] = year
  stats_df = stats_df.append(df)
stats_df['Player'] = stats_df['Player'].str.strip()
stats_df['PlayerName'] = stats_df['Player'].str.split('\\').str[0]
stats_df['PlayerID'] = stats_df['Player'].str.split('\\').str[1]
stats_df.to_csv('data/all_season_stats.csv', index=False)