from __future__ import print_function

import datetime

import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

from hmmlearn.hmm import GaussianHMM
import pandas as pd


print(__doc__)



quotes = pd.read_csv('yahoofinance-INTC.csv',
                     index_col=0,
                     parse_dates=True,
                     infer_datetime_format=True)

date1 = "1995-01-01"
date_med = "2002-06-08"
date2 = "2012-01-06"


train = quotes[(quotes.index >= date1) & (quotes.index <= date_med)]
test = quotes[(quotes.index >= date_med) & (quotes.index <= date2)]

train_dates = train.index
train_close_v = train['Close']
train_volume = train['Volume']

train_diff = np.diff(train_close_v)
train_dates = train_dates[1:]
train_close_v = train_close_v[1:]
train_volume = train_volume[1:]

train_X = np.column_stack([train_diff, train_volume])

test_dates = test.index
test_close_v = test['Close']
test_volume = test['Volume']

test_diff = np.diff(test_close_v)
test_dates = test_dates[1:]
test_close_v = test_close_v[1:]
test_volume = test_volume[1:]

test_X = np.column_stack([test_diff, test_volume])






#Run HMM ##
print("fitting to HMM and decoding ...", end="")

# Make an HMM instance and execute fit
model = GaussianHMM(n_components=5, covariance_type="diag", n_iter=1000).fit(train_X)

# Predict the optimal sequence of internal hidden state
hidden_state_train = model.predict(train_X)
hidden_state_test = model.predict(test_X)

print("done")




print("Transition matrix")
print(model.transmat_)
print()

print("Means and vars of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
    print()

# fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
# colours = cm.rainbow(np.linspace(0, 1, model.n_components))
# for i, (ax, colour) in enumerate(zip(axs, colours)):
#     # Use fancy indexing to plot data in each state.
#     mask = hidden_states == i
#     ax.plot_date(dates[mask], close_v[mask], ".-", c=colour)
#     ax.set_title("{0}th hidden state".format(i))

#     # Format the ticks.
#     ax.xaxis.set_major_locator(YearLocator())
#     ax.xaxis.set_minor_locator(MonthLocator())

#     ax.grid(True)

# plt.show()
N_train = train_X.shape[0]
N_test = test_X.shape[0]
pred_train = np.zeros(N_train)
pred_test = np.zeros(N_test)

for i in range(len(hidden_state_train)):
	state = hidden_state_train[i]
	mean = model.means_[state]
	cov = model.covars_[state]
	change = np.random.multivariate_normal(mean, cov)
	pred_train[i] = change[0]

for i in range(len(hidden_state_test)):
	state = hidden_state_test[i]
	mean = model.means_[state]
	cov = model.covars_[state]
	change = np.random.multivariate_normal(mean, cov)
	pred_test[i] = change[0]

pred_train = np.insert(pred_train, 0, train['Close'][0])
pred_test = np.insert(pred_test, 0, test['Close'][0])
pred_test_price = np.cumsum(pred_test)
pred_train_price = np.cumsum(pred_train)


plt.plot_date(train.index, train['Close'], "-", c="red", label='True')
plt.plot_date(train.index, pred_train_price, "-", c="green", label = 'Predict')
plt.legend()
plt.title('Training period')


plt.plot_date(test.index, test['Close'], "-", c="red",label='True')
plt.plot_date(test.index, pred_test_price, "-", c="green",label = 'Predict')
plt.legend()
plt.title("Test period")





