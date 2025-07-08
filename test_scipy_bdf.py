import numpy as np
from scipy.integrate import solve_ivp
import math

def exponential_decay(t, y):
    return [-y[0]]

# Test parameters
t_span = (0, 1)
y0 = [1.0]
rtol = 1e-3
atol = 1e-6

print("=== SciPy BDF Test ===")
print("Testing dy/dt = -y, y(0) = 1")
print(f"Analytical solution at t=1: {math.exp(-1.0)}")
print()

# Test with BDF method
sol = solve_ivp(exponential_decay, t_span, y0, method='BDF', rtol=rtol, atol=atol)

print(f"SciPy BDF result: {sol.y[0][-1]}")
print(f"Expected:         {math.exp(-1.0)}")
print(f"Error:            {abs(sol.y[0][-1] - math.exp(-1.0))}")
print()

# Also test with different tolerances
print("Testing with different tolerances:")
for rtol_test, atol_test in [(1e-6, 1e-9), (1e-9, 1e-12)]:
    sol = solve_ivp(exponential_decay, t_span, y0, method='BDF', rtol=rtol_test, atol=atol_test)
    print(f"rtol={rtol_test}, atol={atol_test}: {sol.y[0][-1]}, error={abs(sol.y[0][-1] - math.exp(-1.0))}")
