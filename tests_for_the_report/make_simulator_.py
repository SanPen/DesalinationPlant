from Engine.SimpleCalculationEngine import *


def make_simulator(file_name='data.xls',
                   solar=[100, 10000], solar_cost=1500,
                   wind=[100, 10000], wind_cost=1500,
                   battery=[100, 10000], battery_cost=1500,
                   land_cost=100000):
    """

    :param file_name:
    :param solar:
    :param solar_cost:
    :param wind:
    :param wind_cost:
    :param battery:
    :param battery_cost:
    :param land_cost:
    :return:
    """

    profiles = DataProfiles(file_name=file_name)

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
                           solar_power_min=solar[0],
                           solar_power_max=solar[1],
                           unitary_cost=solar_cost,
                           maintenance_perc_cost=0.20,
                           life=25,
                           after_life_investment=0.8,
                           land_usage_rate=3.358e-4,
                           land_cost=land_cost)
    solar_farm.nominal_power = 5000

    wind_farm = WindFarm(profile=profiles.wind_speed,
                         wt_curve_df=profiles.wind_turbine_curve,
                         wind_power_min=wind[0],
                         wind_power_max=wind[1],
                         unitary_cost=wind_cost,
                         maintenance_perc_cost=0.30,
                         life=25,
                         after_life_investment=0.8,
                         land_usage_rate=2.5e-6,
                         land_cost=land_cost)
    wind_farm.nominal_power = 1000

    battery = BatterySystem(charge_efficiency=0.9, discharge_efficiency=0.9, max_soc=0.99, min_soc=0.3,
                            battery_energy_min=battery[0], battery_energy_max=battery[1], unitary_cost=battery_cost,
                            maintenance_perc_cost=0.2, life=7, after_life_investment=1,
                            land_usage_rate=0,
                            land_cost=land_cost)
    battery.nominal_energy = 10000

    sim = Simulator(plant=plant, solar=solar_farm, wind=wind_farm, battery=battery,
                    profiles=profiles)

    return sim
