# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: mind4-minm4-benchmarking
#     language: python
#     name: mind4-minm4-benchmarking
# ---

# %%
# Necessary packages.
import numpy as np
from numpy.linalg import *
import pandas as pd
import warnings
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('always', UserWarning) 
np.set_printoptions(suppress=True)
pd.set_option("display.float_format", "{:.0f}".format)

# Packages only for plotting.
import matplotlib as plt

# Importing the functions from the .py-files with the same name respectively.
from mind4 import mind4
from minm4 import minm4

# %% [markdown]
# ## Input Data

# %%
m_df = pd.read_csv('monthly_data.csv', index_col=0)
y_df = pd.read_csv('yearly_data.csv', index_col=0)

m_df.index = pd.PeriodIndex(m_df.index, freq='M')
y_df.index = pd.PeriodIndex(y_df.index, freq='A')

# %% [markdown]
# ## Value to be Benchmarked

# %%
(m_df.resample('Y').sum()-y_df)

# %% [markdown]
# ### Defining Parameters

# %%
list_to_benchmarking = y_df.columns.to_list()

baseyear = 2022
firstyear = 2016

# %% [markdown]
# # Benchmarking

# %%
result_d4 = mind4(m_df, y_df, list_to_benchmarking, baseyear, firstyear)

#result

# %%
result_m4 = minm4(m_df, y_df, list_to_benchmarking, baseyear, firstyear)

#result_m4

# %% [markdown]
# ## Effect of Benchmarking

# %%
(result_d4/m_df).plot()

# %%
(result_m4/m_df).plot()

# %% [markdown]
# # Deviations on the Total Post Benchmarking.

# %%
# With MinD4.
(result_d4.resample('Y').sum()-y_df)

# %%
# With MinD4.
(result_m4.resample('Y').sum()-y_df)

# %%