import pandas as pd

from alphainspect.utils import cumulative_returns

df = pd.DataFrame({'A': [1.1] * 10, 'B': [1.2] * 10})
df1 = df.cumprod(axis=0)
print(df1)
print(df1.mean(axis=1))

rr = df.copy()
df[:] = 0.5

x = cumulative_returns(rr, df, period=10)
print(x)
