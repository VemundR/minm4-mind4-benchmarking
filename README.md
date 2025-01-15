# mind4-minm4-benchmarking
---
This is a fully flegded Python version of the MinD4 and MinM4 benchmarking methods to use on monthly series against yearly series. These methods spring from the protportional Dentom benchmarking technique. The MinM4 benchmarking method minimizes the proportional, least square difference in relative adjustment to neighbouring quarters, subject to the constraint that the sum of the monthly values for a given year sum up to the corresponding yearly benchmark value. The MinD4 method does the same, but further takes into account the monthly values leading and following the years respectively. Both result in smooth benchmarking effects within the year. In addition MinD4 provides smoothing across the current, leading and following years so that the method doesn't cause jumps in the time series between years.

For further documentation of the proportional Denton method, see chapter VI, C and annex 6.3. of the IMF manual: https://www.imf.org/external/pubs/ft/qna/2000/textbook/ch6.pdf

The following is the math behind 

From Annex 6.A3.1.: 

The first-order conditions for a minimum of the proportional Denton adjustment
formula can be found with the help of the following Lagrange-function:
```math
L(X_1, \dots, X_4) = \sum_{t=2}^{4y} \left[ \frac{X_t}{I_t} - \frac{X_{t-1}}{I_{t-1}} \right]^2 
+ 2 \lambda_y \left[ \sum_{t=4y-3}^{4y} X_t - A_y \right]
```
s.t.
```math
\sum_{t=4y-3}^{4y} X_t = A_y
```

where $\ t \in \{1, \dots, (4\beta), \dots, T\}\$ and $\ y \in \{1, \dots, \beta\}\$.

The first-order conditions of the objective function is as follows:
```math
\frac{\delta L}{\delta X_1} = \frac{1}{I_1^2} X_1 - \frac{1}{I_1 I_2} X_2 + \lambda_1 = 0,
```
```math
\frac{\delta L}{\delta X_2} = -\frac{1}{I_1 I_2} X_1 + \frac{2}{I_2^2} X_2 - \frac{1}{I_2 I_3} X_3 + \lambda_1 = 0,
```
```math
\vdots
```
```math
\frac{\delta L}{\delta X_5} = -\frac{1}{I_4 I_5} X_4 + \frac{2}{I_5^2} X_5 - \frac{1}{I_5 I_6} X_6 + \lambda_2 = 0,
```
```math
\vdots
```
```math
\frac{\delta L}{\delta X_t} = -\frac{1}{I_{t-1} I_t} X_{t-1} + \frac{2}{I_t^2} X_t - \frac{1}{I_t I_{t+1}} X_{t+1} + \lambda_y = 0.
```

Together with the optimization constraints, this constitutes a system of linear equations which can be written in matrix notation as $\ IX=A\$. The matrices are as follows:

```math
I =
\begin{bmatrix}
    \frac{1}{I_1^2} & -\frac{1}{I_1 I_2} & 0 & 0 & 0 & 0 & 0 & 0 & \big| & 1 & 0 \\
    -\frac{1}{I_1 I_2} & \frac{2}{I_2^2} & -\frac{1}{I_2 I_3} & 0 & 0 & 0 & 0 & 0 & \big| & 1 & 0 \\
    0 & -\frac{1}{I_2 I_3} & \frac{2}{I_3^2} & -\frac{1}{I_3 I_4} & 0 & 0 & 0 & 0 & \big| & 1 & 0 \\
    0 & 0 & -\frac{1}{I_3 I_4} & \frac{2}{I_4^2} & -\frac{1}{I_4 I_5} & 0 & 0 & 0 & \big| & 1 & 0 \\
    0 & 0 & 0 & -\frac{1}{I_4 I_5} & \frac{2}{I_5^2} & -\frac{1}{I_5 I_6} & 0 & 0 & \big| & 0 & 1 \\
    0 & 0 & 0 & 0 & -\frac{1}{I_5 I_6} & \frac{2}{I_6^2} & -\frac{1}{I_6 I_7} & 0 & \big| & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & -\frac{1}{I_6 I_7} & \frac{2}{I_7^2} & -\frac{1}{I_7 I_8} & \big| & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0 & -\frac{1}{I_7 I_8} & \frac{2}{I_8^2} & \big| & 0 & 1 \\
    - & - & - & - & - & - & - & - & \big| & - & - \\
    1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & \big| & 0 & 0 \\
    0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 & \big| & 0 & 0 \\
\end{bmatrix},

X =
\begin{bmatrix}
    X_1 \\ X_2 \\ X_3 \\ X_4 \\ X_5 \\ X_6 \\ X_7 \\ X_8 \\ \lambda_1 \\ \lambda_2
\end{bmatrix},

\quad
A =
\begin{bmatrix}
    0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ A_1 \\ A_2
\end{bmatrix}.
```

To use this method, simply utilise the functions minm4(m_df, y_df, list, baseyear, startyear) or mind4(m_df, y_df, list, baseyear, startyear). These functions take the arguments:

m_df: a dataframe with a monthly pd.PeriodIndex, 

y_df: a dataframe with a yearly pd.PeriodIndex, 

list: a list with names of the series in the dataframes which you wish to benchmark(can also be a string), 

baseyear: the final year you wish to benchmark, 

startyear: the first year which you wish to benchmark.


This will return a pd.DataFrame of the benchmarked series. You can display the ratio of the benchmarked and unbenchmarked series to evaluate the effect of the benchmarking.

Make sure that the series you wish to benchmark exist in both the dataframes and the list, if not you will be warned about this. If there are series you do not wish to benchmark in your dataframes simply leave these out of the list and they will be excluded in the resulting dataframe. 



By:

Vemund <uve@ssb.no>

Magnus Kvåle Helliesen <magnus.helliesen@gmail.com>
