from Engine.SimpleCalculationEngine import *


obj_fun_dict = dict()

'''
LCOE = 0,
COST = 1,
INCOME = 2,
INCOME_COST_RATIO = 3
'''
obj_fun_dict['0_MIN_LCOE'] = OptimizationType.LCOE
obj_fun_dict['1_MIN_COST'] = OptimizationType.COST
obj_fun_dict['2_MAX_INCOME'] = OptimizationType.INCOME
obj_fun_dict['3_MAX_INCOME_COST_RATIO'] = OptimizationType.INCOME_COST_RATIO
########################################################################################################################
# Load standard data
########################################################################################################################
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


wind_farm = WindFarm(profile=profiles.wind_speed,
                     wt_curve_df=profiles.wind_turbine_curve,
                     wind_power_min=100,
                     wind_power_max=10000,
                     unitary_cost=900,
                     maintenance_perc_cost=0.30,
                     life=25,
                     after_life_investment=0.8)


battery = BatterySystem(charge_efficiency=0.9, discharge_efficiency=0.9, max_soc=0.99, min_soc=0.3,
                        battery_energy_min=0, battery_energy_max=100000, unitary_cost=900,
                        maintenance_perc_cost=0.2, life=7, after_life_investment=1)

sim = Simulator(plant=plant, solar=solar_farm, wind=wind_farm, battery=battery, profiles=profiles)


########################################################################################################################
# Test specific values
########################################################################################################################
'''
Cuando selecciono uno que tenga el LCOE de alrededor de 3.5 céntimos (más o menos lo que estoy buscando) sale esto, el coste total negativo

Conditions:

Amortization years:	25
Discount rate:	0.03
Optimizing for:	0_MIN_LCOE

Results:

Solar farm size:	1.00/ best:(2,634.00) kW.
Wind farm size:	1,441.73/ best:(1.72) kW.
Storage size:	29.11/ best:(107.26) kWh.

Solar farm cost:	700.00 €.
Wind farm cost:	1,297,554.36 €.
Storage cost:	26,200.73 €.

Total costs:	-805,169.49 €.
Total income:	8,261,707.80 €.
Average benefit:	348,726.05 €/year.
LCOE:	-0.032699 €/kWh.
'''
solar_farm.nominal_power = 1
wind_farm.nominal_power = 1.2
battery.nominal_energy = 1.9
########################################################################################################################
# run test
########################################################################################################################
f = sim.simulate()
print(sim.results_df)
print(f)

sim.export('issue4_results.xlsx')

txt = sim.text_report(obj_fun_dict=obj_fun_dict)
print(txt)

plt.show()
