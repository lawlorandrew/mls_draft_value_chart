import pandas as pd

years = range(2016,2021)

drafts_df = pd.DataFrame()
for year in years:
  superdraft = pd.read_csv(f'data/superdrafts/{year}_superdraft.csv', delimiter='	')
  superdraft = superdraft.rename(columns={'Pick #': 'Pick', 'P': 'Pick' })
  superdraft['year'] = year
  drafts_df = drafts_df.append(superdraft)

drafts_df['Player'] = drafts_df['Player'].str.strip()
drafts_df['Player'] = drafts_df['Player'].str.replace('*', '')
drafts_df['Player'] = drafts_df['Player'].str.replace('^', '')
# handling dupe player names
drafts_df.loc[drafts_df['Player'] == 'Dominic Oduro', 'Player'] = 'Dominic Oduro 1'
drafts_df.to_csv('data/all_superdrafts.csv', index=False)