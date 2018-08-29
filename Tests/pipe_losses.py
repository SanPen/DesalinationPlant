from Engine.CalculationEngine import Pipe
import numpy as np

pipe = Pipe(D=1.5, k=0.1/1000, L=200, unitary_cost=0, maintenance_perc_cost=0, life=0, after_life_investment=0)

n = 4
Q = np.random.rand(n) * 120

losses_approx = pipe.losses(Q=Q, approx=True)

losses_exact = pipe.losses(Q=Q, approx=False)

print('Q', Q, 'm3/s')
print('exact', losses_exact, 'm')
print('approx', losses_approx, 'm')
print('diff', losses_approx - losses_exact, 'm')
