import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
import numpy as np
from sklearn.metrics import mean_squared_error
import unidecode
import seaborn as sns

salary_df = pd.read_csv('./data/2019-salaries.csv')
print(salary_df.columns)
print(salary_df.shape)
print(salary_df.head())

VALUATION_METRIC = 'Min'

stats_df = pd.read_csv('./data/all_season_stats.csv')
stats_df['PlayerName'] = stats_df['PlayerName'].apply(lambda x: unidecode.unidecode(x))
drafts_df = pd.read_csv('./data/all_superdrafts.csv')
drafts_df = drafts_df[drafts_df['year'] < 2020]
drafts_df['Player'] = drafts_df['Player'].apply(lambda x: unidecode.unidecode(x))
pre_2016_df = pd.read_csv('./data/superdrafts/pre-2016.csv')
pre_2016_df = pre_2016_df.rename(columns={ 'pick': 'Pick', 'yr.one.two.min': 'Min' })
draft_stats_df = pd.merge(drafts_df, stats_df, left_on='Player', right_on='PlayerName', how='left', suffixes=['_draft', '_stats'])
draft_stats_df = draft_stats_df.reset_index()
draft_stats_df = draft_stats_df.fillna(0)
# print(draft_stats_df[draft_stats_df['Pick'] == 67][['PlayerName', 'year_draft', 'Min', 'year_stats']])
# nonmatching = drafts_df[~drafts_df['Player'].isin(draft_stats_df['PlayerName'])]
# print(nonmatching.shape)
# nonmatching[['Player', 'year']].to_csv('./nonmatching.csv')

# print(dupes_df[dupes_df['PlayerName'] > 1])
first_season_draft_stats_df = draft_stats_df.sort_values(by='year_stats', ascending=True)
# first_season_draft_stats_df = first_season_draft_stats_df.drop_duplicates(subset='Player_draft')
first_season_draft_stats_df = first_season_draft_stats_df.groupby(by='Player_draft').head(2)
# print(first_season_draft_stats_df.head())
# dupes_df = first_season_draft_stats_df.groupby(by='Player_draft')['PlayerName'].count()
# print(dupes_df)
# print(first_season_draft_stats_df[first_season_draft_stats_df['year_stats'] < first_season_draft_stats_df['year_draft']][['PlayerName', 'year_draft', 'Min', 'year_stats']])
first_season_draft_stats_df = first_season_draft_stats_df.append(pre_2016_df)
# print(first_season_draft_stats_df[first_season_draft_stats_df['Pick'] == 63][['Player_draft', 'year_draft', 'Min', 'year_stats']])
pick_value_df = first_season_draft_stats_df.groupby(by='Pick')[[VALUATION_METRIC]].mean()
pick_value_df = pick_value_df.reset_index()
pick_value_df = pick_value_df[pick_value_df['Pick'] <= 50]
print(pick_value_df)
bf_params, bf_cov = curve_fit(lambda t,a,b,c: a*np.exp(-b*t)+c,  pick_value_df['Pick'],  pick_value_df[VALUATION_METRIC])
# print(bf_params)
fig, ax = plt.subplots()
log_df = np.log(pick_value_df[VALUATION_METRIC])
log_df[log_df < 0] = 0
print(log_df)
polyfit_coefs = np.polyfit(pick_value_df['Pick'], log_df, 1)
# print(polyfit_coefs)
polyfit_model = np.exp(polyfit_coefs[1]) * np.exp(polyfit_coefs[0] * pick_value_df['Pick'])
# print(polyfit_model)
# ax.plot(pick_value_df['Pick'], polyfit)
print(bf_params)
print(polyfit_model)
print(polyfit_coefs)

pick_value_df['Value'] = bf_params[0]*np.exp(-bf_params[1]*pick_value_df['Pick'])+bf_params[2]
# pick_value_df['Polyfit'] = polyfit_model
first_pick = pick_value_df.iloc[0]
print(first_pick)
pick_value_df[f'${VALUATION_METRIC}_pct'] = pick_value_df[VALUATION_METRIC] / first_pick[VALUATION_METRIC]
pick_value_df['PctValue'] = pick_value_df['Value'] / first_pick['Value']
# pick_value_df['Polyfit'] = pick_value_df['Polyfit'] / first_pick['Polyfit']
pick_value_df.to_csv('./output/pick_values.csv', index=False)
print(pick_value_df)
rmsCurve = mean_squared_error(pick_value_df[VALUATION_METRIC], pick_value_df['Value'], squared=False)
# rmsPolyfit = mean_squared_error(pick_value_df[VALUATION_METRIC], pick_value_df['Polyfit'], squared=False)
print('CURVE RMSE: ', rmsCurve)
# print('POLYFIT RMSE: ', rmsPolyfit)
ax.scatter(x=pick_value_df['Pick'], y=pick_value_df[f'${VALUATION_METRIC}_pct'])
# ax.plot(pick_value_df['Pick'], pick_value_df['Polyfit'], color='black', label='Polyfit')
ax.plot(pick_value_df['Pick'], pick_value_df['PctValue'])
ax.set_ylabel('Pick Value (as percentage of top pick)')
ax.set_xlabel('Pick')
ax.set_title('MLS Draft Pick Values based on Minutes Played in First Two Seasons')
fig.text(
  s='By Andrew Lawlor',
  fontsize=6,
  x=0.99,
  y=0.01,
  ha='right',
  va='bottom'
)
plt.savefig(f'./output/{VALUATION_METRIC} values.png')

first_season_draft_stats_df['MLS team'] = first_season_draft_stats_df['MLS team'].str.split('[').str[0].str.strip()
first_season_draft_stats_df.loc[first_season_draft_stats_df['MLS team'].isnull(), 'MLS team'] = first_season_draft_stats_df.loc[first_season_draft_stats_df['MLS team'].isnull(), 'team']
first_season_draft_stats_df['MLS team'] = first_season_draft_stats_df['MLS team'].map({
  'CHI': 'Chicago Fire',
  'CHV': 'Chivas USA',
  'CLB': 'Columbus Crew SC',
  'COL': 'Colorado Rapids',
  'DC': 'D.C. United',
  'FCD': 'FC Dallas',
  'HOU': 'Houston Dynamo',
  'LA': 'LA Galaxy',
  'MTL': 'Montreal Impact',
  'NE': 'New England Revolution',
  'NYC': 'New York City FC',
  'NYRB': 'New York Red Bulls',
  'ORL': 'Orlando City SC',
  'PHI': 'Philadelphia Union',
  'POR': 'Portland Timbers',
  'RSL': 'Real Salt Lake',
  'SEA': 'Seattle Sounders FC',
  'SJ': 'San Jose Earthquakes',
  'SKC': 'Sporting Kansas City',
  'TOR': 'Toronto FC',
  'VAN': 'Vancouver Whitecaps FC'
}).fillna(first_season_draft_stats_df['MLS team'])
first_season_draft_stats_df['PctValue'] = first_season_draft_stats_df[VALUATION_METRIC] / pick_value_df[VALUATION_METRIC][0]
ev_df = pd.merge(first_season_draft_stats_df, pick_value_df, left_on='Pick', right_on='Pick', suffixes=['_actual', '_pv'], how='outer')
ev_df = ev_df[ev_df['Pick'] <= 50]
print(ev_df.head())
print(pick_value_df[VALUATION_METRIC][0])
# ev_df[['MLS team', 'PctValue_actual', 'Min_actual', 'Min_pv', 'Pick']].to_csv('./test.csv')
print(ev_df['PctValue_actual'])
print(ev_df['PctValue_pv'])
ev_df['Value_Over_Expected'] = ev_df['PctValue_actual'] - ev_df['PctValue_pv']
print(ev_df['Value_Over_Expected'])
# draft_team_df = ev_df.groupby(by='MLS team')[['Value_Over_Expected']].sum()
# print(draft_team_df)
# print(draft_team_df.sum())
draft_team_df = ev_df.groupby(by='MLS team')[['Value_Over_Expected']].mean()
print(ev_df[ev_df['MLS team'] == 'New York City FC'])
draft_team_df['count'] = ev_df.groupby(by='MLS team')[['Value_Over_Expected']].count()
draft_team_df = draft_team_df.sort_values(by='Value_Over_Expected', ascending=False)
print(draft_team_df)
fig, ax = plt.subplots()
xvals = np.arange(draft_team_df.shape[0])
width = 0.8
clist = [(0, "red"), (0.125, "red"), (0.25, "orange"), (0.5, "green"), 
         (0.7, "green"), (0.75, "blue"), (1, "blue")]
rvb = mcolors.LinearSegmentedColormap.from_list("", clist)
# ax.bar(xvals, draft_team_df['Value_Over_Expected'], width, color=rvb(draft_team_df['Value_Over_Expected']))
# ax.set_xticks(xvals)
sns.barplot(ax=ax, x=draft_team_df.index, y=draft_team_df['Value_Over_Expected'], palette="coolwarm_r")
ax.set_xticklabels(draft_team_df.index, rotation=90, fontsize=6)
ax.set_ylabel('Avg. Draft Pick Efficiency')
ax.set_title('MLS Draft Efficiency By Team')
fig.text(
  s='By Andrew Lawlor',
  fontsize=6,
  x=0.99,
  y=0.01,
  ha='right',
  va='bottom'
)
plt.savefig('./output/Draft Teams.png')
# print(draft_team_df.sum())