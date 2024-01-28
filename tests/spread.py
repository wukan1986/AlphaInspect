import pandas as pd

from alphainspect.utils import cumulative_returns

returns = pd.DataFrame({'A': [1.01] * 10, 'B': [1.02] * 10,
                        'C': [1.01] * 10, 'D': [1.02] * 10,
                        })
weights = pd.DataFrame({'A': [0.25] * 10, 'B': [0.25] * 10,
                        'C': [-0.25] * 10, 'D': [-0.25] * 10,
                        })

x1 = cumulative_returns(returns.to_numpy(), weights.to_numpy(), funds=2, freq=5)
print(x1)

weights = pd.DataFrame({'A': [0.5] * 10, 'B': [0.5] * 10,
                        'C': [0] * 10, 'D': [0] * 10,
                        })
x2 = cumulative_returns(returns.to_numpy(), weights.to_numpy(), funds=2, freq=5)
print(x2)

weights = pd.DataFrame({'A': [0] * 10, 'B': [0] * 10,
                        'C': [-0.5] * 10, 'D': [-0.5] * 10,
                        })
x3 = cumulative_returns(returns.to_numpy(), weights.to_numpy(), funds=2, freq=5)
print(x3)
print(x2 + x3)

weights = pd.DataFrame({'A': [0] * 10, 'B': [0] * 10,
                        'C': [0.5] * 10, 'D': [0.5] * 10,
                        })
x4 = cumulative_returns(returns.to_numpy(), weights.to_numpy(), funds=2, freq=5)
print(x4)

print(x2 + (2 - x4))
