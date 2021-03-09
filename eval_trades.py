import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

trade_dollars_df = pd.DataFrame(
  [
    [8, 100000],
    [6, 125000],
    [9, 75000],
    [11, 75000],
    [10, 100000],
    [3, 150000],
    [15, 50000],
    [10, 85000], # 2018
    [7, 150000], # 2018
    [4, 200000], # 2018
    [3, 200000], # 2018 (half TAM half GAM)
    # [70, 50000], # 2018 (this was for an intl spot)
    [26, 37500],
    [27, 37500],
    # 26 & 27 for 75000 in 2017
    [16, 75000], # 2017
    [3, 250000], # 2017
    # [24, 75000] # 2017 for two intl spots, but one in future
  ],
  columns=['Pick', '$']
)
print(trade_dollars_df)
polyfit_coefs = np.polyfit(trade_dollars_df['Pick'], np.log(trade_dollars_df['$']), 1)
xvals = np.linspace(1,50,50)
print(polyfit_coefs)
polyfit_model = np.exp(polyfit_coefs[1]) * np.exp(polyfit_coefs[0] * xvals)
print(polyfit_model)
print(polyfit_model / polyfit_model[0])
print(polyfit_model[4] + polyfit_model[31])
print(polyfit_model[30] + 125000)
fig, ax = plt.subplots()
ax.scatter(trade_dollars_df['Pick'], trade_dollars_df['$'] / 100000)
ax.plot(xvals, polyfit_model / 100000, label='Pick Value Based on Trades')
pv_df = pd.read_csv('./output/pick_values.csv')
ax.plot(pv_df['Pick'], pv_df['Value']*polyfit_model[0]/(pv_df['Value'][0] * 100000), color='black', label='Pick Value Based on Minutes Played, Converted to Allocation Money')
ax.set_ylabel('Allocation Money (in $100k increments)')
ax.set_xlabel('Pick')
ax.legend(fontsize=8)
ax.set_title('MLS Draft Pick Trade Value')
fig.text(
  s='By Andrew Lawlor',
  fontsize=6,
  x=0.99,
  y=0.01,
  ha='right',
  va='bottom'
)
plt.savefig('./output/trade dollars.png')

pv_df['Allocation Money Value'] = pv_df['Value']*polyfit_model[0]/pv_df['Value'][0]
pv_df[['Pick', 'PctValue', 'Allocation Money Value']].to_csv('./output/allocation money values.csv', index=False)

def get_value(pick, pv_df):
  return pv_df['Allocation Money Value'][pick - 1]

print('ATL - DC 2021')
print(get_value(5, pv_df) + get_value(32, pv_df))
print(get_value(31, pv_df) + 125000)
print('HOU - COL 2021')
print(get_value(3, pv_df))
print(get_value(6, pv_df) + 200000)
print('NYC - LA 2019')
print(get_value(19, pv_df) + 75000)
print(get_value(12, pv_df))
print('COL - CHI 2019')
print(get_value(15, pv_df) + 100000)
print(get_value(5, pv_df))
# def trade_func(x, trade):
#   def pv_func(picks, cash):
#     runningsum = 0
#     for p in picks:
#       runningsum += x[0]*np.exp(x[1]*p)
#       # runningsum += np.exp(np.exp(x[0]) + np.exp(x[1] * p))
#     # print(runningsum + cash)
#     # logged_cash = np.log(cash) if cash > 0 else 0
#     # print(logged_cash)
#     return runningsum + cash
#   return [pv_func(trade['team1picks'], trade['team1$']), pv_func(trade['team2picks'], trade['team2$'])]
  
  
# # print(trade_func([5, 32], [31]))
# # print(trade_func([5, 32], [31])[0]([0,0]))
# trade = pd.Series(
#   [[10], 0, [], 100000],
#   index=['team1picks', 'team1$', 'team2picks', 'team2$']
# )
# root = fsolve(trade_func, [12.22994626, -0.09127497], args=trade, factor=1)
# # print(np.exp(root[0]) + np.exp(root[1] * 5))
# # print(root)
# # print(trade_func(root, [[[5, 32], 0], [[31], 125000]]))
# print(root)
# fig, ax = plt.subplots()
# xvals = np.linspace(1, 50, 50)
# yvals = root[0]*np.exp(root[1]*xvals)
# print(yvals[0])
# yvals = yvals / yvals[0]
# ax.plot(xvals, yvals)
# print(yvals)
# print(np.exp(root[0])*np.exp(root[1]*5))
# print(root[0]*np.exp(root[1]*31))
# print(root[0]*np.exp(root[1]*32))
# print(trade_func(root, trade))
# print(root[0]*np.exp(root[1]*5) + root[0]*np.exp(root[1]*32))
# print(root[0]*np.exp(root[1]*31) + 125000)
# plt.savefig('./test.png')



# # def func2(x):
# #   return [x[0]*np.exp(x[1]*1), x[0]*np.exp(x[1]*13) + x[0]*np.exp(x[1]*14)]

# # root = fsolve(func2, [7.62453588, -0.05538672], maxfev=100000)
# # print(root)
# # fig, ax = plt.subplots()
# # x = np.linspace(1, 50, 50)
# # print(np.exp(root[0] + np.exp(root[1] * x)) / np.exp(root[0] + np.exp(root[1] * x))[0])
# # ax.plot(x, np.exp(root[0] + np.exp(root[1] * x)) / np.exp(root[0] + np.exp(root[1] * x))[0])
# # plt.savefig('./test2.png')
# # print([root[0]*np.exp(root[1]*1), root[0]*np.exp(root[1]*13) + root[0]*np.exp(root[1]*14)])