
from itertools import product

import numpy as np
import pandas as pd

from tests_for_the_report.make_simulator_ import make_simulator
from tests_for_the_report.plot_3D import plot_3D


if __name__ == '__main__':

    solar = [1, 10000]
    wind = [1, 10000]
    battery = [1, 10000]

    solar_cost = 500  # https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2018/Jan/IRENA_2017_Power_Costs_2018.pdf
    wind_cost = 850  # source: https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2018/Jan/IRENA_2017_Power_Costs_2018.pdf
    battery_cost = 400  # https://arena.gov.au/blog/arenas-role-commercialising-big-batteries/

    simulator = make_simulator(file_name='data.xls',
                               solar=solar, solar_cost=solar_cost,
                               wind=wind, wind_cost=wind_cost,
                               battery=battery, battery_cost=battery_cost)

    # brute force to save all the values
    num = 25
    s = np.linspace(solar[0], solar[1], num=num)
    w = np.linspace(wind[0], wind[1], num=num)
    b = np.linspace(battery[0], battery[1], num=num)

    X = list()
    Y = list()

    i = 0
    n = len(s) * len(w) * len(b)
    for x in product(s, w, b):
        print(x)

        simulator.set_state(x)
        f, f_all = simulator.simulate()  # returns [f_lcoe, f_cost, f_income, f_income_cost, f_benefit]

        X.append(x)
        Y.append(f_all)

        prog = (i+1) * 100.0 / n
        print(i, n, prog, '%')
        i += 1

    df_X = pd.DataFrame(X, columns=['Solar (kW)', 'Wind (kW)', 'Battery (kWh)'])
    df_Y = pd.DataFrame(Y, columns=['LCOE', 'Cost', 'Income', 'Income_Cost_Ratio', 'Benefit'])

    experiment_name = 'Space_mapping/Space_mapping_num_' + str(num)

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
