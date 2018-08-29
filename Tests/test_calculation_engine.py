import pandas as pd
from Engine.CalculationEngine import *

'''
Pipe roughness values
http://www.pumpfundamentals.com/download-free/pipe_rough_values.pdf
'''

profiles = DataProfiles('data.xls')

membranes = Membranes(membrane_number=2,
                      membrane_power=482,
                      membrane_production=0.2079,
                      unitary_cost=500,
                      maintenance_perc_cost=5,
                      life=10,
                      after_life_investment=1)

# Desalination plant: Deposit
deposit = Deposit(capacity=1e6,
                  head=170,
                  water_demand=profiles.water_demand,
                  investment_cost=1000000,
                  maintenance_perc_cost=0.8,
                  life=50,
                  after_life_investment=1)

# Desalination plant: Sea intake pipe
sea_pipe = Pipe(D=3,
                k=1.5e-6,  # PVC
                L=520,
                unitary_cost=200,
                maintenance_perc_cost=1,
                life=25,
                after_life_investment=1)

# Desalination plant: Storage pipe
sto_pipe = Pipe(D=3,
                k=1.5e-6,  # PVC
                L=5700,
                unitary_cost=200,
                maintenance_perc_cost=1,
                life=25,
                after_life_investment=1)

# Desalination plant: Storage pump
sto_pump = Pump(units=2*3,
                nominal_power=120,
                performance_capacity=0.79,
                unitary_cost=0,
                maintenance_perc_cost=0,
                life=25,
                after_life_investment=1)

# Desalination plant: sea pump
sea_pump = Pump(units=2*3,
                nominal_power=120,
                performance_capacity=0.79,
                unitary_cost=1000,
                maintenance_perc_cost=2,
                life=25,
                after_life_investment=1)

# complete desalination plant
desalination_plant = DesalinationPlant(membranes=membranes,
                                       sea_pump=sea_pump,
                                       sea_pipe=sea_pipe,
                                       sto_pump=sto_pump,
                                       sto_pipe=sto_pipe,
                                       deposit=deposit)


solar_farm = SolarFarm(profile=profiles.solar_irradiation,
                       solar_power_min=100,
                       solar_power_max=10000,
                       unitary_cost=700,
                       maintenance_perc_cost=20,
                       life=25,
                       after_life_investment=0.8)
solar_farm.nominal_power = 500


wind_farm = WindFarm(profile=profiles.wind_speed,
                     wt_curve_df=profiles.wind_turbine_curve,
                     wind_power_min=100,
                     wind_power_max=10000,
                     unitary_cost=900,
                     maintenance_perc_cost=30,
                     life=25,
                     after_life_investment=0.8)
wind_farm.nominal_power = 1000


# Compute the raw production
p = solar_farm.power() + wind_farm.power()

# compute the desalination
total_power, water_mismatch = desalination_plant.process_power(p, profiles.time)

desalination_plant.plot()

plt.show()


