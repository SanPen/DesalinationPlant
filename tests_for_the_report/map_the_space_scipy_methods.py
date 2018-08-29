from scipy.optimize import rosen, differential_evolution, minimize
from stochopy import Evolutionary
from yabox import DE
import pandas as pd
import os

from tests_for_the_report.make_simulator_ import make_simulator, OptimizationType
from tests_for_the_report.plot_3D import plot_3D, plot_2D

if __name__ == '__main__':

    print('Mapping with DYCORS...')
    solar = [1, 10000]
    wind = [1, 10000]
    battery = [1, 10000]
    num = 5000
    solar_cost = 500  #  https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2018/Jan/IRENA_2017_Power_Costs_2018.pdf
    wind_cost = 850  #  source: https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2018/Jan/IRENA_2017_Power_Costs_2018.pdf
    battery_cost = 400  # https://arena.gov.au/blog/arenas-role-commercialising-big-batteries/
    bounds = [solar, wind, battery]


    def callback(arg, convergence):
        print(convergence, arg)


    methods = [
               'differential_evolution',
               'BFGS',
               'GA',
               'DE'
              ]

    for method in methods:

        print(method)
        simulator = make_simulator(file_name='data.xls',
                                   solar=solar, solar_cost=solar_cost,
                                   wind=wind, wind_cost=wind_cost,
                                   battery=battery, battery_cost=battery_cost)

        simulator.optimization_type = OptimizationType.LCOE

        n = len(bounds)
        upper = [None] * n
        lower = [None] * n
        x0 = [None] * n

        for i in range(n):
            upper[i] = bounds[i][1]
            lower[i] = bounds[i][0]
            x0[i] = lower[i] + (upper[i] - lower[i]) / 2.0

        if method == 'differential_evolution':

            # https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.differential_evolution.html

            result = differential_evolution(simulator.objfunction, bounds, strategy='best1bin', maxiter=num,
                                            popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
                                            callback=callback, disp=False, polish=True, init='latinhypercube')

        elif method == 'BFGS':

            res = minimize(simulator.objfunction, x0, args=(), method='BFGS', jac=None, hess=None, hessp=None,
                           bounds=bounds, constraints=(), tol=None, callback=None, options=None)

        elif method == 'DE':

            ea = DE(simulator.objfunction, bounds, mutation=(0.5, 1.0), maxiters=num)
            p, f = ea.solve(show_progress=True)

        elif method == 'GA':

            ea = Evolutionary(simulator.objfunction, lower, upper, popsize=15, max_iter=num, random_state=3)
            xopt, gfit = ea.optimize(solver="cpso")

        # the simulator object stores all the data itself
        df_X = pd.DataFrame(simulator.X, columns=['Solar (kW)', 'Wind (kW)', 'Battery (kWh)'])
        df_Y = pd.DataFrame(simulator.Y, columns=['LCOE', 'Cost', 'Income', 'Income_Cost_Ratio', 'Benefit'])

        experiment_name = 'Optimization_mapping/' + method + '/Space_mapping_' + method + '_' + str(num)

        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        print('Saving...')
        writer = pd.ExcelWriter(experiment_name + '.xlsx')
        df_X.to_excel(writer, sheet_name='X')
        df_Y.to_excel(writer, sheet_name='Y')
        writer.save()
        print('Done!')

        # plot 3D
        for obj in ['LCOE', 'Cost', 'Income', 'Income_Cost_Ratio', 'Benefit']:
            plot_3D(experiment_name, df_X, df_Y, x_serie='Solar (kW)', y_serie='Wind (kW)', z_serie=obj)
            plot_3D(experiment_name, df_X, df_Y, x_serie='Wind (kW)', y_serie='Battery (kWh)', z_serie=obj)
            plot_3D(experiment_name, df_X, df_Y, x_serie='Solar (kW)', y_serie='Battery (kWh)', z_serie=obj)

            plot_2D(experiment_name, df_X, df_Y, z_serie=obj)
