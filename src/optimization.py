from scipy.optimize import linprog
import numpy as np

def optimize_inventory(forecasted_demand, holding_cost, shortage_cost, initial_inventory=0):
    """
    Perform inventory optimization using linear programming.
    """
    n_periods = len(forecasted_demand)
    c = holding_cost * np.ones(n_periods)  # Cost function (holding cost)
    
    A_eq = np.zeros((n_periods, n_periods))
    b_eq = np.zeros(n_periods)
    
    for i in range(n_periods):
        A_eq[i, :i+1] = 1  # Cumulative sum constraint
        b_eq[i] = forecasted_demand[:i+1].sum() - initial_inventory

    bounds = [(0, None) for _ in range(n_periods)]  # Inventory levels can't be negative
    
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.success:
        print("Optimization Successful!")
        return result.x  # Optimal inventory levels
    else:
        print("Optimization Failed!")
        return None

