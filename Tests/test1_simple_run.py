from Engine.SimpleCalculationEngine import *
import time

profiles = DataProfiles('data.xls')

osmosis = SimpleReverseOsmosisSystem(units=6,
                                     aux_power=59 / 3,
                                     nominal_unit_flow_in=280.16 / 3,
                                     nominal_unit_flow_out=100.2 / 3,
                                     unitary_cost=1,
                                     maintenance_perc_cost=0.18,
                                     life=8,
                                     after_life_investment=1.0)

pumps = SimplePump(units=osmosis.units,
                   nominal_power=423 / 3,
                   nominal_flow=285 / 3,
                   unitary_cost=1,
                   maintenance_perc_cost=0.05,
                   life=25,
                   after_life_investment=1.0)

deposit = Deposit(capacity=180000,
                  head=190,
                  water_demand=profiles.water_demand,
                  investment_cost=450000,
                  maintenance_perc_cost=0.01,
                  life=50,
                  after_life_investment=0.2)

plant = SimpleDesalinationPlant(pumps=pumps, deposit=deposit, reverse_osmosis_system=osmosis)

solar_farm = SolarFarm(profile=profiles.solar_irradiation,
                       solar_power_min=100,
                       solar_power_max=10000,
                       unitary_cost=700,
                       maintenance_perc_cost=0.20,
                       life=25,
                       after_life_investment=0.8)
solar_farm.nominal_power = 5000

wind_farm = WindFarm(profile=profiles.wind_speed,
                     wt_curve_df=profiles.wind_turbine_curve,
                     wind_power_min=100,
                     wind_power_max=10000,
                     unitary_cost=900,
                     maintenance_perc_cost=0.30,
                     life=25,
                     after_life_investment=0.8)
wind_farm.nominal_power = 1000

battery = BatterySystem(charge_efficiency=0.9, discharge_efficiency=0.9, max_soc=0.99, min_soc=0.3,
                        battery_energy_min=0, battery_energy_max=100000, unitary_cost=900,
                        maintenance_perc_cost=0.2, life=7, after_life_investment=1)
battery.nominal_energy = 10000


sim = Simulator(plant=plant, solar=solar_farm, wind=wind_farm, battery=battery, profiles=profiles)
# sim = Simulator(plant=plant, solar=solar_farm, wind=wind_farm, battery=None, profiles=profiles)


start_time = time.time()
f = sim.simulate()
print("--- %s seconds ---" % (time.time() - start_time))

# print(sim.results_df)
# print(f)
# sim.results_df.to_excel('simple_res.xlsx')

# print('lcoe:', sim.lcoe_calc(20, 0.03), '€/kWh')

# sim.export('full_res.xlsx')

plt.show()
