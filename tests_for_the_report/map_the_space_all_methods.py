from skopt import gp_minimize, forest_minimize, gbrt_minimize

import pandas as pd
import os
from tests_for_the_report.make_simulator_ import make_simulator, OptimizationType
from tests_for_the_report.plot_3D import plot_3D

if __name__ == '__main__':

    print('Mapping with ...')
    solar = [100, 10000]
    wind = [100, 10000]
    battery = [100, 10000]
    num = 5000
    solar_cost = 500  #  https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2018/Jan/IRENA_2017_Power_Costs_2018.pdf
    wind_cost = 850  #  source: https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2018/Jan/IRENA_2017_Power_Costs_2018.pdf
    battery_cost = 400  # https://arena.gov.au/blog/arenas-role-commercialising-big-batteries/
    dimensions = [solar, wind, battery]

    # for method in ['Gradient', 'RandomForest', 'DYCORS', 'GradientBoostTree', 'GaussianProcess']:
    # methods = ['Gradient', 'RandomForest', 'GradientBoostTree', 'GaussianProcess']
    methods = ['GradientBoostTree', 'GaussianProcess']
    for method in methods:

        print(method)
        simulator = make_simulator(file_name='data.xls',
                                   solar=solar, solar_cost=solar_cost,
                                   wind=wind, wind_cost=wind_cost,
                                   battery=battery, battery_cost=battery_cost)

        simulator.optimization_type = OptimizationType.BENEFIT

        if method == 'Gradient':
            gp_minimize(simulator.objfunction, dimensions=dimensions,
                        base_estimator=None, n_calls=num, n_random_starts=10,
                        acq_func="gp_hedge", acq_optimizer="auto", x0=None, y0=None,
                        random_state=None, verbose=False, callback=None, n_points=num, n_restarts_optimizer=5,
                        xi=0.01, kappa=1.96, noise="gaussian", n_jobs=1)

        elif method == 'RandomForest':

            forest_minimize(simulator.objfunction, dimensions=dimensions,
                            base_estimator='ET', n_calls=num, n_random_starts=10, acq_func='EI', x0=None,
                            y0=None, random_state=None, verbose=False, callback=None, n_points=num, xi=0.01, kappa=1.96,
                            n_jobs=1)

        elif method == 'GradientBoostTree':

            gbrt_minimize(simulator.objfunction, dimensions=dimensions,
                          base_estimator=None, n_calls=num, n_random_starts=10, acq_func='EI',
                          acq_optimizer='auto', x0=None, y0=None, random_state=None, verbose=False, callback=None,
                          n_points=num, xi=0.01, kappa=1.96, n_jobs=1)

        elif method == 'GaussianProcess':

            gp_minimize(simulator.objfunction, dimensions=dimensions,
                        base_estimator=None, n_calls=num, n_random_starts=10, acq_func='gp_hedge',
                        acq_optimizer='auto', x0=None, y0=None, random_state=None, verbose=False, callback=None,
                        n_points=num, n_restarts_optimizer=5, xi=0.01, kappa=1.96, noise='gaussian', n_jobs=1)

        elif method == 'DYCORS':
            simulator.max_eval = num
            simulator.run()

        df_X = pd.DataFrame(simulator.X, columns=['Solar (kW)', 'Wind (kW)', 'Battery (kWh)'])
        df_Y = pd.DataFrame(simulator.Y, columns=['LCOE', 'Cost', 'Income', 'Income_Cost_Ratio', 'Benefit'])

        print(simulator.solution)
        # simulator.set_state(simulator.solution)
        # simulator.simulate(25)

        obj_fun_dict = dict()
        obj_fun_dict['0_BENEFIT'] = OptimizationType.BENEFIT
        obj_fun_dict['1_MIN_LCOE'] = OptimizationType.LCOE
        obj_fun_dict['2_MIN_COST'] = OptimizationType.COST
        obj_fun_dict['3_MAX_INCOME'] = OptimizationType.INCOME
        obj_fun_dict['4_MAX_INCOME_COST_RATIO'] = OptimizationType.INCOME_COST_RATIO
        print(simulator.text_report(obj_fun_dict))

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
