import pandas as pd
import os
import numpy as np
from enum import Enum
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from PyQt5.QtCore import QThread, pyqtSignal
from pySOT import *
from poap.controller import SerialController
from warnings import warn

from numba import jit, f8, bool_, int_, autojit
from numpy import arange

plt.style.use('ggplot')


class ObjectiveFunctionType(Enum):
    LCOE = 0,
    GridUsage = 1,
    GridUsageCost = 2,
    GridUsageCost_times_LCOE = 3


class DataProfiles:

    time = np.zeros(0)

    solar_irradiation = np.zeros(0)

    wind_speed = np.zeros(0)

    wind_direction = np.zeros(0)

    normalized_electric_demand = np.zeros(0)

    water_demand = np.zeros(0)

    spot_price = np.zeros(0)

    secondary_reserve_price = np.zeros(0)

    wind_turbine_curve = np.zeros(0)

    def __init__(self, file_name):
        """

        :param file_name:
        """

        '''
        time
        solar_irradiation (MW/m2)
        wind_speed (m/s):60
        wind_direction(º)
        normalized_electric_demand
        water_demand
        secondary_reg_price
        spot_price
        water price
        '''

        if len(file_name) > 0 and os.path.exists(file_name):
            # load file
            xl = pd.ExcelFile(file_name)

            # assert that the requires sheets exist. This sort of determines if the excel file is the right one
            c1 = 'profiles' in xl.sheet_names
            c2 = 'AG_CAT' in xl.sheet_names

            cond = c1 and c2

            if cond:

                df = xl.parse(sheet_name='profiles')

                # Magic procedure to assign the profiles to the variables in this class
                def get_array(key):
                    for col in df.columns.values:
                        if key in col:
                            return df[col]

                self.solar_irradiation = get_array('radia').values

                self.wind_speed = get_array('speed').values

                self.wind_direction = get_array('direc').values

                self.normalized_electric_demand = get_array('elect').values

                self.water_demand = get_array('water_demand').values

                self.water_price = get_array('water_price').values

                self.spot_price = get_array('spot').values

                self.secondary_reserve_price = get_array('secon').values

                self.time = get_array('tim')

                # load the wind turbine power curve and normalize it
                self.wind_turbine_curve = xl.parse(sheet_name='AG_CAT')['P (kW)']

            xl.close()

        else:
            warn('The file ' + file_name + ' does not exists.')


class GeneralDevice:

    def __init__(self, investment_cost, maintenance_perc_cost, life, after_life_investment):
        """

        :param unitary_cost:
        :param maintenance_perc_cost:
        :param life:
        :param after_life_investment:
        """
        self.investment_cost = investment_cost

        self.maintenance_perc_cost = maintenance_perc_cost

        self.life = life

        self.after_life_investment = after_life_investment


class Membranes(GeneralDevice):

    def __init__(self, membrane_number, membrane_power, membrane_production, unitary_cost, maintenance_perc_cost,
                 life, after_life_investment):
        """
        Membranes aggregation
        :param membrane_number:
        :param membrane_power:
        :param membrane_production:
        :param unitary_cost:
        :param maintenance_perc_cost:
        :param life:
        :param after_life_investment:
        """
        GeneralDevice.__init__(self, unitary_cost*membrane_number, maintenance_perc_cost, life, after_life_investment)

        self.membrane_number = membrane_number

        self.membrane_power = membrane_power

        self.membrane_production = membrane_production

        self.power_arr = None

        self.water_arr = None

    def process_power(self, P):
        """
        Given an amount of available power P
        compute how much of that power is used and the water produced by the membrane system
        :param P: power in kW
        :return: power used by the membrane system (kW), water produced (m3/s)
        """

        operative_membranes = np.floor(P / self.membrane_power)

        n = len(P)
        operative_membranes = np.ones(n) * operative_membranes

        idx = np.where(operative_membranes > self.membrane_number)[0]
        operative_membranes[idx] = self.membrane_number

        self.power_arr = operative_membranes * self.membrane_power

        self.water_arr = self.power_arr * self.membrane_production

        return self.power_arr, self.water_arr

    def plot(self, ax=None, title='Membranes'):
        """
        plot
        :param ax: MatPlotLib axis
        :param title
        :return:
        """
        if ax is None:
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            ax = plt.gca()
            ax2 = ax.twinx()
        ax.plot(self.water_arr, label='water produced', color='r')
        ax2.plot(self.power_arr, label='power used', color='b', alpha=0.5)
        ax.set_xlabel('time')
        ax.set_ylabel('Water produced (m3/s)')
        ax2.set_ylabel('Power used (kW)')
        ax.set_title(title)
        ax.legend()
        ax2.legend()

    def get_cost(self, N):
        """
        Number of years
        :param N:
        :return:
        """

        investment = self.investment_cost * self.units

        maintenance = self.maintenance_perc_cost * investment

        arr = np.zeros(N + 1)
        arr[0] = investment
        arr[1:] = maintenance

        # re-buy
        inc = int(self.life)
        for idx in range(inc, N, inc):
            arr[idx] += investment * self.after_life_investment  # incur percentage of the investment after life

        return arr


@autojit(nopython=True)
def process_deposit_water(Qin, water_demand, capacity, SoC_0, water, water_flow, water_mismatch, SoC):
    """
    Process the deposit water
    :param Qin: Water flow coming in (m3/s)
    :param time: time array
    :return:
    """

    # Compute the water flow balance (Supply - Demand)
    n = len(Qin)

    # Compute the water
    # water = np.zeros(n)
    # water_mismatch = np.zeros(n)

    water[0] = capacity * SoC_0
    SoC[0] = SoC_0

    for i in range(1, n):

        # compute the water balance
        dt = 1  # (time[i] - time[i-1]).seconds
        w = water[i-1] + Qin[i - 1] * dt - water_demand[i-1]

        if w > capacity:
            # print('spill water')
            water[i] = capacity
            water_mismatch[i] = w - capacity
        elif w < 0:
            # print('empty')
            water[i] = 0
            water_mismatch[i] = w
        else:
            # ok
            water[i] = w
            water_mismatch[i] = 0

        # Compute the state of charge
        SoC[i] = water[i] / capacity

        water_flow[i] = water[i] - water[i-1]



class Deposit(GeneralDevice):

    def __init__(self, capacity, head, water_demand, investment_cost, maintenance_perc_cost, life,
                 after_life_investment):
        """
        Deposit object
        :param capacity:
        :param head:
        :param unitary_cost:
        :param maintenance_perc_cost:
        :param life:
        :param after_life_investment:
        """
        GeneralDevice.__init__(self, investment_cost, maintenance_perc_cost, life, after_life_investment)

        self.capacity = capacity

        self.head = head

        self.water_demand = water_demand

        self.SoC_0 = 0.5

        self.water = None

        self.water_flow = None

        self.water_mismatch = None

        self.SoC = None

    def process_water_(self, Qin, time):
        """
        Process the deposit water
        :param Qin: Water flow coming in (m3/s)
        :param time: time array
        :return:
        """

        # Compute the water flow balance (Supply - Demand)
        n = len(Qin)

        # Compute the water
        self.water = np.zeros(n)
        self.water_flow = np.zeros(n)
        self.water_mismatch = np.zeros(n)

        self.water[0] = self.capacity * self.SoC_0

        for i in range(1, n):

            # compute the water balance
            dt = 1  # (time[i] - time[i-1]).seconds
            w = self.water[i-1] + Qin[i - 1] * dt - self.water_demand[i-1]

            if w > self.capacity:
                # print('spill water')
                self.water[i] = self.capacity
                self.water_mismatch[i] = w - self.capacity
            elif w < 0:
                # print('empty')
                self.water[i] = 0
                self.water_mismatch[i] = w
            else:
                # ok
                self.water[i] = w
                self.water_mismatch[i] = 0

        # Compute the state of charge
        self.SoC = self.water / self.capacity

        self.water_flow = np.r_[np.diff(self.water), 0]

        return self.water_mismatch

    def process_water(self, Qin, time):
        """
        Process the deposit water
        :param Qin: Water flow coming in (m3/s)
        :param time: time array
        :return:
        """
        n = len(Qin)

        self.water = np.zeros(n)
        self.water_flow = np.zeros(n)
        self.water_mismatch = np.zeros(n)
        self.SoC = np.zeros(n)

        process_deposit_water(Qin, self.water_demand, self.capacity, self.SoC_0,
                              self.water, self.water_flow, self.water_mismatch, self.SoC)

        return self.water_mismatch

    def plot(self, ax=None, title='Deposit'):
        """
        plot
        :param ax: MatPlotLib axis
        :return:
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.water, label='water stored')
        ax.plot(self.water_mismatch, label='mismatch')
        ax.plot(self.water_demand, label='demand')
        ax.plot(self.water_flow, label='water flow')
        ax.set_xlabel('time')
        ax.set_ylabel('Water (m3)')
        ax.set_title(title)
        ax.legend()

    def get_cost(self, N):
        """
        Number of years
        :param N:
        :return:
        """

        investment = self.investment_cost

        maintenance = self.maintenance_perc_cost * investment

        arr = np.zeros(N + 1)
        arr[0] = investment
        arr[1:] = maintenance

        # re-buy
        inc = int(self.life)
        if inc > 0:
            for idx in range(inc, N, inc):
                arr[idx] += investment * self.after_life_investment  # incur percentage of the investment after life
        return arr


class Pump(GeneralDevice):

    def __init__(self, units, nominal_power, performance_capacity, unitary_cost, maintenance_perc_cost, life,
                 after_life_investment):
        """
        Pump object
        :param units:
        :param nominal_power:
        :param performance_capacity: variation of the performance with the flow Q (per unit/m3/s)
        :param unitary_cost:
        :param maintenance_perc_cost:
        :param life:
        :param after_life_investment:
        """
        GeneralDevice.__init__(self, unitary_cost*units, maintenance_perc_cost, life, after_life_investment)

        self.units = units

        self.nominal_power = nominal_power

        self.performance_capacity = performance_capacity

        self.power_arr = None

        self.performance = None

    def power(self, Q, H, ro=1000.0, g=9.81):
        """
        Returns the power consumed by the pump(s)
        :param Q: Flow (m3/s)
        :param H: Head (pipe head + pipe losses)
        :param ro: water density (kg/m3)
        :param g: gravity acceleration (m/s2)
        :return: The pump(s) power in kW
        """

        idx = np.where(Q > 0)[0]
        n = len(Q)

        # compute the associated performance
        self.performance = (Q[idx] / self.units) * self.performance_capacity

        # power in watts (kW)
        self.power_arr = np.zeros(n)
        self.power_arr[idx] = Q[idx] * H[idx] * ro * g / (self.performance * 1000.0)

        return self.power_arr

    def plot(self, ax=None, title='pump'):
        """
        plot
        :param ax: MatPlotLib axis
        :return:
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.power_arr, label='power')
        ax.plot(self.performance * 100, label='performance')
        ax.set_xlabel('time')
        ax.set_ylabel('Pump power kW')
        ax.set_title(title)
        ax.legend()

    def get_cost(self, N):
        """
        Number of years
        :param N:
        :return:
        """

        investment = self.investment_cost * self.units

        maintenance = self.maintenance_perc_cost * investment

        arr = np.zeros(N + 1)
        arr[0] = investment
        arr[1:] = maintenance

        # re-buy
        inc = int(self.life)
        for idx in range(inc, N, inc):
            arr[idx] += investment * self.after_life_investment  # incur percentage of the investment after life

        return arr


class Pipe(GeneralDevice):

    def __init__(self, D, k, L, unitary_cost, maintenance_perc_cost, life, after_life_investment):
        """
        Pipe object
        :param D:
        :param k:
        :param L:
        :param unitary_cost:
        :param maintenance_perc_cost:
        :param life:
        :param after_life_investment:
        """
        GeneralDevice.__init__(self, unitary_cost, maintenance_perc_cost, life, after_life_investment)

        self.D = D

        self.k = k

        self.L = L

        self.losses_arr = None

    def losses(self, Q, ro=1000.0, mu=0.001, g=9.81, eps=1e-9, approx=True):
        """
        Solve the pipe losses using Newton Raphson
        :param Q: Array of fluid flows (m3/s)
        :param ro: Density (kg/m3)
        :param mu: Viscosity (kg/(s·m))
        :param g: gravity acceleration (m/s2)
        :param eps: Numerical zero
        :return: The pipe loses in m
        """

        idx = np.where(Q > 0)[0]
        n = len(Q)
        # Reynolds number (only for flows greater then zero)
        Re = (4 * ro * Q[idx]) / (mu * np.pi * self.D)

        if approx:
            # Serghides's approximation
            # https://en.wikipedia.org/wiki/Darcy_friction_factor_formulae
            m = self.k / self.D / 3.7
            A = -2 * np.log10(m + 12 / Re)
            B = -2 * np.log10(m + 2.51 * A / Re)
            C = -2 * np.log10(m + 2.51 * B / Re)

            val = A - ((B - A) * (B - A)) / (C - 2 * B + A)
            friction = np.zeros(n)
            friction[idx] = 1 / (val * val)

        else:

            def fx(x):
                """
                Colebrook function
                :param x: Value of friction
                :return:
                """
                return (1 / np.sqrt(x)) + 2 * np.log10(((self.k / self.D) / 3.7) + (2.51 / (Re * np.sqrt(x))))

            def dfx(x):
                """
                Colebrook function derivative
                :param x: Value of friction
                :return:
                """
                return -1 / (2 * np.power(x, 1.5)) - 251 / (100 * np.power(x, 1.5) * np.log(10) *
                       ((10 * self.k) / (37 * self.D) + 251 / (100 * np.sqrt(x) * Re)) * Re)

            friction0 = eps
            friction = friction0 - fx(friction0) / dfx(friction0)
            mismatch = sum(abs(friction - friction0))

            while mismatch > eps:
                friction0 = friction
                friction = friction0 - fx(friction0) / dfx(friction0)
                mismatch = sum(abs(friction - friction0))

        # Compute the losses given the computed friction (Darcy-Weissbach)
        v = 4 * Q / (self.D * self.D * np.pi)
        self.losses_arr = friction * self.L * v * v / (2 * self.D * g)

        return self.losses_arr

    def plot(self, ax=None, title='Pipe'):
        """
        plot
        :param ax: MatPlotLib axis
        :return:
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(self.losses_arr)
        ax.set_xlabel('time')
        ax.set_ylabel('Pipe losses (m)')
        ax.set_title(title)


class Grid(GeneralDevice):

    def __init__(self, connection_power, spot_price, secondary_price, unitary_cost, maintenance_perc_cost, life,
                 after_life_investment):
        """
        Pipe object
        :param D:
        :param k:
        :param L:
        :param unitary_cost:
        :param maintenance_perc_cost:
        :param life:
        :param after_life_investment:
        """
        GeneralDevice.__init__(self, unitary_cost, maintenance_perc_cost, life, after_life_investment)

        self.connection_power = connection_power

        self.spot_price = spot_price

        self.secondary_price = secondary_price


class DesalinationPlant:

    def __init__(self, membranes: Membranes, sea_pump: Pump, sea_pipe: Pipe, sto_pump: Pump, sto_pipe: Pipe,
                 deposit: Deposit, head=0):
        """
        Objects aggregation to conform the desalination plant
        :param membranes:
        :param sea_pump:
        :param sea_pipe:
        :param sto_pump:
        :param sto_pipe:
        :param deposit:
        :param head:
        """

        self.head = head

        self.membranes = membranes

        self.sea_pump = sea_pump

        self.sea_pipe = sea_pipe

        self.sto_pump = sto_pump

        self.sto_pipe = sto_pipe

        self.deposit = deposit

    def process_power(self, res_power, time):
        """
        Process the desalination plant production with RES
        :param res_power: array of power values
        :param time: matching time stamps array
        :return:
        """
        # power used by the membranes and water produced
        membrane_power, Q = self.membranes.process_power(res_power)

        # compute the pumping heads and nominal head + pipe losses
        H_sea = self.head + self.sea_pipe.losses(Q)
        H_sto = self.deposit.head + self.sto_pipe.losses(Q)

        # pumps power
        sea_pump_power = self.sea_pump.power(Q, H_sea)
        sto_pump_power = self.sto_pump.power(Q, H_sto)

        total_power = membrane_power + sea_pump_power + sto_pump_power

        # dt = (time[i] - time[i - 1]).seconds

        water_mismatch = self.deposit.process_water(Q, time)

        return total_power, water_mismatch

    def plot(self):
        """

        :return:
        """
        self.membranes.plot(title='Membranes')

        self.sea_pipe.plot(title='Sea pipe')
        self.sto_pipe.plot(title='Storage pipe')

        self.sea_pump.plot(title='Sea pump')
        self.sto_pump.plot(title='Storage pump')

        self.deposit.plot(title='Storage Deposit')


class SolarFarm(GeneralDevice):

    def __init__(self, profile, solar_power_min=0, solar_power_max=10000, unitary_cost=200,
                 maintenance_perc_cost=0, life=25, after_life_investment=0, land_usage_rate=0, land_cost=0):
        """

        Args:
            profile: Solar horizontal irradiation profile (W/m2) [1D array]
            solar_power_max: Maximum power in kW to consider when sizing
            unitary_cost: Cost peer installed kW of the solar facility (€/kW)
        """

        GeneralDevice.__init__(self, unitary_cost, maintenance_perc_cost, life, after_life_investment)

        self.index = None

        self.nominal_power = None

        self.irradiation = profile

        self.normalized_power = profile / 1000.0

        self.max_power = solar_power_max

        self.min_power = solar_power_min

        self.unitary_cost = unitary_cost

        self.land_usage_rate = land_usage_rate

        self.land_cost = land_cost

    def power(self):
        """
        Returns the generated power
        Returns: Array
        """
        return self.nominal_power * self.normalized_power

    def cost(self):
        """

        :return:
        """
        return self.unitary_cost * self.nominal_power

    def get_cost(self, N):
        """
        Number of years
        :param N:
        :return:
        """

        land = self.land_usage_rate * self.nominal_power * self.land_cost

        investment = self.investment_cost * self.nominal_power + land

        maintenance = self.maintenance_perc_cost * investment

        arr = np.zeros(N + 1)
        arr[0] = investment
        arr[1:] = maintenance

        # re-buy
        inc = int(self.life+1)
        if inc > 0:
            for idx in range(inc, N, inc):
                arr[idx] += investment * self.after_life_investment  # incur percentage of the investment after life
        return arr


class WindFarm(GeneralDevice):

    def __init__(self, profile, wt_curve_df, wind_power_min=0, wind_power_max=10000, unitary_cost=900,
                 maintenance_perc_cost=0, life=25, after_life_investment=0, land_usage_rate=0, land_cost=0):
        """

        Args:
            profile: Wind profile in m/s
            wt_curve_df: Wind turbine power curve in a DataFrame (Power [any unit, values] vs. Wind speed [m/s, index])
            wind_power_max: Maximum nominal power of the wind park considered when sizing
            unitary_cost: Unitary cost of the wind park in €/kW

            Example of unitary cost:
            A wind park with 4 turbines of 660 kW cost 2 400 000 €
            2400000 / (4 * 660) = 909 €/kW installed
        """
        GeneralDevice.__init__(self, unitary_cost, maintenance_perc_cost, life, after_life_investment)

        self.index = None

        self.nominal_power = None

        self.wt_curve_df = wt_curve_df

        # load the wind turbine power curve and normalize it
        ag_curve = interp1d(wt_curve_df.index, wt_curve_df.values / wt_curve_df.values.max())

        self.wind_speed = profile

        self.normalized_power = ag_curve(profile)

        self.max_power = wind_power_max

        self.min_power = wind_power_min

        self.unitary_cost = unitary_cost

        self.land_usage_rate = land_usage_rate

        self.land_cost = land_cost

    def power(self):
        """
        Returns the generated power
        Returns: Array
        """
        return self.nominal_power * self.normalized_power

    def cost(self):
        """

        :return:
        """
        return self.unitary_cost * self.nominal_power

    def get_cost(self, N):
        """
        Number of years
        :param N:
        :return:
        """

        land = self.land_usage_rate * self.nominal_power * self.land_cost

        investment = self.investment_cost * self.nominal_power + land

        maintenance = self.maintenance_perc_cost * investment

        arr = np.zeros(N + 1)
        arr[0] = investment
        arr[1:] = maintenance

        # re-buy
        inc = int(self.life+1)
        if inc > 0:
            for idx in range(inc, N, inc):
                arr[idx] += investment * self.after_life_investment  # incur percentage of the investment after life
        return arr



# @jit(arg_types=[f8[:], f8, int_[:], f8,
#                 f8, f8, f8,
#                 f8, f8, f8, bool_,
#                 f8[:], f8[:], f8[:], f8[:], f8[:], int_[:]])
@autojit(nopython=True)
def simulate_storage_power_array(P_array, soc_0, time_array, nominal_energy,
                                 discharge_efficiency, charge_efficiency, charge_per_cycle,
                                 min_soc_charge, min_soc, max_soc, charge_if_needed,
                                 r_demanded_power, r_energy, r_power, r_grid_power, r_soc):
    """
    The storage signs are the following

    supply power: positive
    recharge power: negative

    this means that a negative power will ask the battery to charge and
    a positive power will ask the battery to discharge

    to match these signs to the give profiles, we should invert the
    profiles sign
    Args:
        P_array: Power array that is sent to the battery [Negative charge, positive discharge]
        soc_0: State of charge at the beginning [0~1]
        time_array: Array of seconds
        charge_if_needed: Allow the battery to take extra power that is not given in P. This limits the growth of
        the battery system in the optimization since the bigger the battery, the more grid power it will take to
        charge when RES cannot cope. Hence, since we're minimizing the grid usage, there is an optimum battery size
    Returns:
        energy: Energy effectively processed by the battery
        power: Power effectively processed by the battery
        grid_power: Power dumped array
        soc: Battery state of charge array
    """

    if nominal_energy is None:
        raise Exception('You need to set the battery nominal power!')

    # initialize arrays
    # P_array = np.array(P_array)
    nt = len(P_array)

    energy = np.zeros(nt + 1)
    power = np.zeros(nt + 1)
    soc = np.zeros(nt + 1)
    grid_power = np.zeros(nt + 1)

    energy[0] = nominal_energy * soc_0
    soc[0] = soc_0

    charge_energy_per_cycle = nominal_energy * charge_per_cycle

    for t in range(nt-1):

        # if np.isnan(P_array[t]):
        #     warn('NaN found!!!!!!')

        # pick the right efficiency value
        if P_array[t] >= 0:
            eff = discharge_efficiency
        else:
            eff = charge_efficiency

        # the time comes in nanoseconds, we need the time step in hours
        dt = (time_array[t + 1] - time_array[t]) / 3600.0

        # compute the proposed energy. Later we check how much is actually possible
        proposed_energy = energy[t] - P_array[t] * dt * eff

        # charge the battery from the grid if the SoC is too low and we are allowing this behaviour
        if charge_if_needed and soc[t] < min_soc_charge:
            proposed_energy -= charge_energy_per_cycle / dt  # negative is for charging

        # Check the proposed energy
        if proposed_energy > nominal_energy * max_soc:  # Truncated, too high

            energy[t + 1] = nominal_energy * max_soc
            power[t + 1] = (energy[t]-energy[t + 1]) / (dt * eff)
            grid_power[t + 1] = power[t + 1] + P_array[t]

        elif proposed_energy < nominal_energy * min_soc:  # Truncated, too low

            energy[t + 1] = nominal_energy * min_soc
            power[t + 1] = (energy[t]-energy[t + 1]) / (dt * eff)
            grid_power[t + 1] = - power[t + 1] + P_array[t]

        else:  # everything is within boundaries

            energy[t + 1] = proposed_energy
            power[t + 1] = P_array[t]
            grid_power[t + 1] = 0

        # Update the state of charge
        soc[t + 1] = energy[t + 1] / nominal_energy

        # Assign a results
        r_demanded_power[t+1] = P_array[t]
        r_energy[t] = energy[t]
        r_power[t] = power[t]
        r_grid_power[t] = grid_power[t]
        r_soc[t] = soc[t]

    # assign the last value
    t = nt-1
    r_demanded_power[t] = P_array[t]
    r_energy[t] = energy[t]
    r_power[t] = power[t]
    r_grid_power[t] = grid_power[t]
    r_soc[t] = soc[t]


class BatterySystem(GeneralDevice):

    def __init__(self, charge_efficiency=0.9, discharge_efficiency=0.9, max_soc=0.99, min_soc=0.3,
                 battery_energy_min=0, battery_energy_max=100000, unitary_cost=900,
                 maintenance_perc_cost=0, life=7, after_life_investment=0, land_usage_rate=0, land_cost=0):
        """

        Args:
            charge_efficiency: Efficiency when charging
            discharge_efficiency:  Efficiency when discharging
            max_soc: Maximum state of charge
            min_soc: Minimum state of charge
            battery_energy_max: Maximum energy in kWh allowed for sizing the battery
            unitary_cost: Cost per kWh of the battery (€/kWh)
        """
        GeneralDevice.__init__(self, unitary_cost, maintenance_perc_cost, life, after_life_investment)

        self.demanded_power = None

        self.energy = None

        self.power = None

        self.grid_power = None

        self.soc = None

        self.time = None

        self.index = None

        self.nominal_energy = None

        self.charge_efficiency = charge_efficiency

        self.discharge_efficiency = discharge_efficiency

        self.max_soc = max_soc

        self.min_soc = min_soc

        self.min_soc_charge = (self.max_soc + self.min_soc) / 2  # SoC state to force the battery charge

        self.charge_per_cycle = 0.1  # charge 10% per cycle

        self.max_energy = battery_energy_max

        self.min_energy = battery_energy_min

        self.unitary_cost = unitary_cost

        self.land_usage_rate  = land_usage_rate

        self.land_cost = land_cost

        self.results = None

    def simulate_array_(self, P, soc_0, time, charge_if_needed=False):
        """
        The storage signs are the following

        supply power: positive
        recharge power: negative

        this means that a negative power will ask the battery to charge and
        a positive power will ask the battery to discharge

        to match these signs to the give profiles, we should invert the
        profiles sign
        Args:
            P: Power array that is sent to the battery [Negative charge, positive discharge]
            soc_0: State of charge at the beginning [0~1]
            time: Array of DataTime values
            charge_if_needed: Allow the battery to take extra power that is not given in P. This limits the growth of
            the battery system in the optimization since the bigger the battery, the more grid power it will take to
            charge when RES cannot cope. Hence, since we're minimizing the grid usage, there is an optimum battery size
        Returns:
            energy: Energy effectively processed by the battery
            power: Power effectively processed by the battery
            grid_power: Power dumped array
            soc: Battery state of charge array
        """

        if self.nominal_energy is None:
            raise Exception('You need to set the battery nominal power!')

        # initialize arrays
        P = np.array(P)
        nt = len(P)
        energy = np.zeros(nt + 1)
        power = np.zeros(nt + 1)
        soc = np.zeros(nt + 1)
        grid_power = np.zeros(nt + 1)
        energy[0] = self.nominal_energy * soc_0
        soc[0] = soc_0

        charge_energy_per_cycle = self.nominal_energy * self.charge_per_cycle

        for t in range(nt-1):

            if np.isnan(P[t]):
                warn('NaN found!!!!!!')

            # pick the right efficiency value
            if P[t] >= 0:
                eff = self.discharge_efficiency
            else:
                eff = self.charge_efficiency

            # the time comes in nanoseconds, we need the time step in hours
            dt = (time[t + 1] - time[t]).seconds / 3600

            # compute the proposed energy. Later we check how much is actually possible
            proposed_energy = energy[t] - P[t] * dt * eff

            # charge the battery from the grid if the SoC is too low and we are allowing this behaviour
            if charge_if_needed and soc[t] < self.min_soc_charge:
                proposed_energy -= charge_energy_per_cycle / dt  # negative is for charging

            # Check the proposed energy
            if proposed_energy > self.nominal_energy * self.max_soc:  # Truncated, too high

                energy[t + 1] = self.nominal_energy * self.max_soc
                power[t + 1] = (energy[t]-energy[t + 1]) / (dt * eff)
                grid_power[t + 1] = power[t + 1] + P[t]

            elif proposed_energy < self.nominal_energy * self.min_soc:  # Truncated, too low

                energy[t + 1] = self.nominal_energy * self.min_soc
                power[t + 1] = (energy[t]-energy[t + 1]) / (dt * eff)
                grid_power[t + 1] = - power[t + 1] + P[t]

            else:  # everything is within boundaries

                energy[t + 1] = proposed_energy
                power[t + 1] = P[t]
                grid_power[t + 1] = 0

            # Update the state of charge
            soc[t + 1] = energy[t + 1] / self.nominal_energy

        # Compose a results DataFrame
        self.demanded_power = np.r_[0, P[:-1]]
        self.energy = energy[:-1]
        self.power = power[:-1]
        self.grid_power = grid_power[:-1]
        self.soc = soc[:-1]
        self.time = time

        # d = np.c_[np.r_[0, P[:-1]], power[:-1], grid_power[:-1], energy[:-1], soc[:-1] * 100]
        # cols = ['P request', 'P', 'grid', 'E', 'SoC']
        # self.results = pd.DataFrame(data=d, columns=cols)

        return energy[:-1], power[:-1], grid_power[:-1], soc[:-1]

    def simulate_array(self, P, soc_0, time, charge_if_needed=False):
        """
        The storage signs are the following

        supply power: positive
        recharge power: negative

        this means that a negative power will ask the battery to charge and
        a positive power will ask the battery to discharge

        to match these signs to the give profiles, we should invert the
        profiles sign
        Args:
            P: Power array that is sent to the battery [Negative charge, positive discharge]
            soc_0: State of charge at the beginning [0~1]
            time: Array of DataTime values
            charge_if_needed: Allow the battery to take extra power that is not given in P. This limits the growth of
            the battery system in the optimization since the bigger the battery, the more grid power it will take to
            charge when RES cannot cope. Hence, since we're minimizing the grid usage, there is an optimum battery size
        Returns:
            energy: Energy effectively processed by the battery
            power: Power effectively processed by the battery
            grid_power: Power dumped array
            soc: Battery state of charge array
        """

        if self.nominal_energy is None:
            raise Exception('You need to set the battery nominal power!')

        T = (time.values.astype(float) * 1e-9).astype(int)

        nt = len(P)

        self.demanded_power = np.zeros(nt+1)
        self.energy = np.zeros(nt)
        self.power = np.zeros(nt)
        self.grid_power = np.zeros(nt)
        self.soc = np.zeros(nt)

        simulate_storage_power_array(P_array=P,
                                     soc_0=soc_0,
                                     time_array=T,
                                     nominal_energy=self.nominal_energy,
                                     discharge_efficiency=self.discharge_efficiency,
                                     charge_efficiency=self.charge_efficiency,
                                     charge_per_cycle=self.charge_per_cycle,
                                     min_soc_charge=self.min_soc_charge,
                                     min_soc=self.min_soc,
                                     max_soc=self.max_soc,
                                     charge_if_needed=charge_if_needed,
                                     r_demanded_power=self.demanded_power,
                                     r_energy=self.energy,
                                     r_power=self.power,
                                     r_grid_power=self.grid_power,
                                     r_soc=self.soc)

        self.time = time

        return self.energy, self.power, self.grid_power, self.soc

    def plot(self, fig=None):

        if fig is None:
            fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(111)
        ax1.stackplot(self.time, self.power, self.grid_power)
        ax1.plot(self.time, self.demanded_power, linewidth=4)
        ax1.set_ylabel('kW')

        ax2 = ax1.twinx()
        ax2.plot(self.time, self.soc, color='k')
        ax2.set_ylabel('SoC')

        ax1.legend()
        plt.show()

    def get_cost(self, N):
        """
        Number of years
        :param N:
        :return:
        """

        land = self.land_usage_rate * self.nominal_energy * self.land_cost

        investment = self.investment_cost * self.nominal_energy + land

        maintenance = self.maintenance_perc_cost * investment

        arr = np.zeros(N + 1)
        arr[0] = investment
        arr[1:] = maintenance

        # re-buy
        inc = int(self.life+1)
        for idx in range(inc, N, inc):
            arr[idx] += investment * self.after_life_investment  # incur percentage of the investment after life

        return arr


class MicroGrid(QThread):

    progress_signal = pyqtSignal(float)
    progress_text_signal = pyqtSignal(str)
    done_signal = pyqtSignal()

    dim = 3  # 3 variables to optimize

    def __init__(self,
                 time_arr,
                 desalination_plant: DesalinationPlant,
                 grid_connection: Grid,
                 solar_farm: SolarFarm,
                 wind_farm: WindFarm,
                 battery_system: BatterySystem,
                 obj_fun_type: ObjectiveFunctionType=ObjectiveFunctionType.GridUsage,
                 max_eval=100):
        """

        :param time_arr:
        :param desalinatin_plant:
        :param grid_connection:
        :param solar_farm:
        :param wind_farm:
        :param battery_system:
        :param obj_fun_type:
        :param max_eval:
        """

        QThread.__init__(self)

        # variables for the optimization
        self.xlow = np.zeros(self.dim)  # lower bounds
        self.xup = np.array([solar_farm.max_power, wind_farm.max_power, battery_system.max_energy])
        self.info = "Microgrid with Wind turbines, Photovoltaic panels and storage coupled to a demand"  # info
        self.integer = np.array([0])  # integer variables
        self.continuous = np.arange(1, self.dim)  # continuous variables

        # assign the device list
        self.solar_farm = solar_farm

        self.wind_farm = wind_farm

        self.desalination_plant = desalination_plant

        self.battery_system = battery_system

        self.grid = grid_connection

        self.obj_fun_type = obj_fun_type

        # create a time index matching the length
        self.time = time_arr

        # Results

        self.aggregated_demand_profile = None

        self.solar_power_profile = None

        self.wind_power_profile = None

        self.grid_power = None

        self.Energy = None
        self.battery_output_power = None
        self.battery_output_current = None
        self.battery_voltage = None
        self.battery_losses = None
        self.battery_state_of_charge = None
        self.iteration = None
        self.max_eval = max_eval

        self.optimization_values = None
        self.raw_results = None
        self.solution = None

        self.grid_energy = None
        self.energy_cost = None
        self.investment_cost = None
        self.lcoe_val = None

        self.x_fx = list()

    def __call__(self, x, verbose=False):
        """
        Call for this object, performs the dispatch given a vector x of facility sizes
        Args:
            x: vector [solar nominal power, wind nominal power, storage nominal power]

        Returns: Value of the objective function for the given x vector

        """
        return self.objfunction(x, verbose)

    def objfunction(self, x, verbose=False):
        """

        Args:
            x: optimsation vector [solar nominal power, wind nominal power, storage nominal power]

        Returns:

        """
        ################################################################################################################
        # Set the devices nominal power
        ################################################################################################################
        self.solar_farm.nominal_power = x[0]

        self.wind_farm.nominal_power = x[1]

        self.battery_system.nominal_energy = x[2]

        '''
        The profiles sign as given are:

            demand: negative
            generation: positive
        '''

        ################################################################################################################
        # Compute the battery desired profile
        ################################################################################################################

        self.aggregated_demand_profile = self.demand_system.power() + self.wind_farm.power() + self.solar_farm.power()

        ################################################################################################################
        # Compute the battery real profile: processing the desired profile
        ################################################################################################################

        '''
        The storage signs are the following

        supply power: positive
        recharge power: negative

        this means that a negative power will ask the battery to charge and
        a positive power will ask the battery to discharge

        to match these signs to the give profiles, we should invert the
        profiles sign
        '''

        # reverse the power profile sign (the surplus, should go in the battery)
        demanded_power = - self.aggregated_demand_profile
        # initial state of charge
        SoC0 = 0.5

        # calculate the battery values: process the desired power
        # energy, power, grid_power, soc
        self.Energy, \
        self.battery_output_power, \
        self.grid_power, \
        self.battery_state_of_charge = self.battery_system.simulate_array(P=demanded_power, soc_0=SoC0,
                                                                          time=self.time, charge_if_needed=True)

        # the processed values are 1 value shorter since we have worked with time increments

        # calculate the grid power as the difference of the battery power
        # and the profile required for perfect auto-consumption
        # self.grid_power = demanded_power - self.battery_output_power

        # compute the investment cost
        self.investment_cost = self.solar_farm.cost() + self.wind_farm.cost() + self.battery_system.cost()

        # compute the LCOE Levelized Cost Of Electricity
        res = self.lcoe(generated_power_profile=self.grid_power, investment_cost=self.investment_cost,
                        discount_rate=self.investment_rate, years=self.lcoe_years, verbose=verbose)
        self.grid_energy = res[0]
        self.energy_cost = res[1]
        self.investment_cost = res[2]
        self.lcoe_val = res[3]

        # select the objective function
        if self.obj_fun_type is ObjectiveFunctionType.LCOE:
            fx = abs(self.lcoe_val)
        elif self.obj_fun_type is ObjectiveFunctionType.GridUsage:
            fx = sum(abs(self.grid_power))
        elif self.obj_fun_type is ObjectiveFunctionType.GridUsageCost:
            fx = sum(abs(self.grid_power * self.spot_price))
        elif self.obj_fun_type is ObjectiveFunctionType.GridUsageCost_times_LCOE:
            fx = sum(abs(self.grid_power * self.spot_price)) * (1-abs(self.lcoe_val))
        else:
            fx = 0

        # store the values
        self.raw_results.append([fx] + list(x) + [self.lcoe_val, sum(abs(self.grid_power)), self.investment_cost])

        self.iteration += 1
        prog = self.iteration / self.max_eval
        # print('progress:', prog)
        self.progress_signal.emit(prog * 100)

        return fx

    def lcoe(self, generated_power_profile, investment_cost, discount_rate, years, verbose=False):
        """

        :param generated_power_profile:
        :param investment_cost:
        :param discount_rate:
        :param verbose:
        :return:
        """
        grid_energy = generated_power_profile.sum()
        energy_cost = (generated_power_profile * self.spot_price).sum()

        # build the arrays for the n years
        I = np.zeros(years)  # investment
        I[0] = investment_cost
        E = np.ones(years) * grid_energy  # gains/cost of electricity
        M = np.ones(years) * investment_cost * 0.1  # cost of maintenance

        dr = np.array([(1 + discount_rate)**(i+1) for i in range(years)])
        A = (I + M / dr).sum()
        B = (E / dr).sum()

        if verbose:
            print('Grid energy', grid_energy, 'kWh')
            print('Energy cost', energy_cost, '€')
            print('investment_cost', investment_cost, '€')
            print('dr', dr)
            print('A:', A, 'B:', B)
            print('lcoe_val', A/B)

        # self.grid_energy = grid_energy
        # self.energy_cost = energy_cost
        # self.investment_cost = investment_cost
        lcoe_val = A / B

        return grid_energy, energy_cost, investment_cost, lcoe_val

    def economic_sensitivity(self, years_arr, inv_rate_arr):
        """

        :param years_arr:
        :param inv_rate_arr:
        :return:
        """
        ny = len(years_arr)
        ni = len(inv_rate_arr)
        values = np.zeros((4, ni, ny))

        for i, years in enumerate(years_arr):
            for j, inv_rate in enumerate(inv_rate_arr):

                grid_energy, \
                energy_cost, \
                investment_cost, \
                lcoe_val = self.lcoe(generated_power_profile=self.grid_power,
                                     investment_cost=self.investment_cost,
                                     discount_rate=inv_rate,
                                     years=years,
                                     verbose=False)

                values[:, j, i] = np.array([grid_energy, energy_cost, investment_cost, lcoe_val])

        df_lcoe = pd.DataFrame(data=values[3, :, :], index=inv_rate_arr, columns=years_arr)

        return df_lcoe

    def optimize(self):
        self.run()
        
    def run(self):
        """
        Function that optimizes a MicroGrid Object
        Args:

        Returns:

        """
        self.iteration = 0
        self.raw_results = list()

        self.progress_signal.emit(0)
        self.progress_text_signal.emit('Optimizing facility sizes by surrogate optimization...')

        # (1) Optimization problem
        # print(data.info)

        # (2) Experimental design
        # Use a symmetric Latin hypercube with 2d + 1 samples
        exp_des = SymmetricLatinHypercube(dim=self.dim, npts=2 * self.dim + 1)

        # (3) Surrogate model
        # Use a cubic RBF interpolant with a linear tail
        surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=self.max_eval)

        # (4) Adaptive sampling
        # Use DYCORS with 100d candidate points
        adapt_samp = CandidateDYCORS(data=self, numcand=100 * self.dim)

        # Use the serial controller (uses only one thread)
        controller = SerialController(self.objfunction)

        # (5) Use the synchronous strategy without non-bound constraints
        strategy = SyncStrategyNoConstraints(
            worker_id=0, data=self, maxeval=self.max_eval, nsamples=1,
            exp_design=exp_des, response_surface=surrogate,
            sampling_method=adapt_samp)
        controller.strategy = strategy

        # Run the optimization strategy
        result = controller.run()

        # Print the final result
        print('Best value found: {0}'.format(result.value))
        print('Best solution found: {0}'.format(
            np.array_str(result.params[0], max_line_width=np.inf,
                         precision=5, suppress_small=True)))
        self.solution = result.params[0]

        # Extract function values from the controller
        self.optimization_values = np.array([o.value for o in controller.fevals])

        # turn the results into a DataFrame
        data = np.array(self.raw_results)
        self.x_fx = pd.DataFrame(data=data[:, 1:], index=data[:, 0], columns=['Solar', 'Wind', 'Battery', 'LCOE', 'Grid energy', 'Investment'])
        self.x_fx.sort_index(inplace=True)

        self.progress_text_signal.emit('Done!')
        self.done_signal.emit()

    def plot(self):
        """
        Plot the dispatch values
        Returns:

        """
        # plot results
        plot_cols = 3
        plot_rows = 2

        steps_number = len(self.demand_system.normalized_power)

        plt.figure(figsize=(16, 10))
        plt.subplot(plot_rows, plot_cols, 1)
        plt.plot(self.demand_system.power(), label="Demand Power")
        plt.plot(self.solar_farm.power(), label="Photovoltaic Power")
        plt.plot(self.wind_farm.power(), label="Wind Power")
        plt.ylabel('kW')
        plt.legend()

        plt.subplot(plot_rows, plot_cols, 2)
        plt.plot(self.aggregated_demand_profile, label="Aggregated power profile")
        plt.plot(np.zeros(steps_number), 'k')
        plt.ylabel('kW')
        plt.legend()

        plt.subplot(plot_rows, plot_cols, 3)
        plt.plot(- self.aggregated_demand_profile, label="Power demanded to the battery")
        plt.plot(np.zeros(steps_number), 'k')
        plt.ylabel('kW')
        plt.legend()

        plt.subplot(plot_rows, plot_cols, 4)
        plt.plot(self.grid_power, label="Power demanded to the grid")
        plt.plot(np.zeros(steps_number), 'k')
        plt.ylabel('kW')
        plt.legend()

        plt.subplot(plot_rows, plot_cols, 5)
        plt.plot(self.battery_output_power, label="Battery power")
        plt.plot(np.zeros(steps_number), 'k')
        plt.ylabel('kW')
        plt.legend()

        plt.subplot(plot_rows, plot_cols, 6)
        plt.plot(self.battery_state_of_charge, label="Battery SoC")
        plt.plot(np.zeros(steps_number), 'k')
        plt.ylabel('Per unit')
        plt.legend()

    def plot_optimization(self, ax=None):
        """
        Plot the optimization convergence
        Returns:

        """
        if self.optimization_values is not None:
            max_eval = len(self.optimization_values)

            if ax is None:
                f, ax = plt.subplots()
            # Points
            ax.plot(np.arange(0, max_eval), self.optimization_values, 'bo')
            # Best value found
            ax.plot(np.arange(0, max_eval), np.minimum.accumulate(self.optimization_values), 'r-', linewidth=3.0)
            ax.set_xlabel('Evaluations')
            ax.set_ylabel('Function Value')
            ax.set_title('Optimization convergence')

    def export(self, file_name):
        """
        Export definition and results to excel
        :param file_name:
        :return:
        """

        writer = pd.ExcelWriter(file_name)

        # Solar irradiation
        pd.DataFrame(data=self.solar_farm.irradiation,
                     index=self.time,
                     columns=['irradiation (MW/m2)']).to_excel(writer, 'irradiation')

        # wind speed
        pd.DataFrame(data=self.wind_farm.wind_speed,
                     index=self.time,
                     columns=['VEL(m/s):60']).to_excel(writer, 'wind')

        # AG curve
        self.wind_farm.wt_curve_df.to_excel(writer, 'AG_CAT')

        # demand
        pd.DataFrame(data=self.demand_system.normalized_power * -1,
                     index=self.time,
                     columns=['normalized_demand']).to_excel(writer, 'demand')

        # prices
        pd.DataFrame(data=np.c_[self.spot_price, self.band_price],
                     index=self.time,
                     columns=['Secondary_reg_price', 'Spot_price']).to_excel(writer, 'prices')

        # Results
        self.x_fx.to_excel(writer, 'results')

        writer.save()


if __name__ == '__main__':
    print()
    # Create devices
    # fname = 'data.xls'
    #
    # prices = pd.read_excel(fname, sheet_name='prices')[['Secondary_reg_price', 'Spot_price']].values
    #
    # # load the solar irradiation in W/M2 and convert it to kW
    # solar_radiation_profile = pd.read_excel(fname, sheet_name='irradiation')['irradiation (MW/m2)'].values
    #
    # #  create the solar farm object
    # solar_farm = SolarFarm(solar_radiation_profile)
    #
    # # Load the wind speed in m/s
    # wind_speed_profile = pd.read_excel(fname, sheet_name='wind')['VEL(m/s):60'].values
    #
    # # load the wind turbine power curve and normalize it
    # ag_curve_df = pd.read_excel(fname, sheet_name='AG_CAT')['P (kW)']
    #
    # # create the wind farm object
    # wind_farm = WindFarm(wind_speed_profile, ag_curve_df)
    #
    # # load the demand values and set it negative for the sign convention
    # demand_profile = pd.read_excel(fname, sheet_name='demand')['normalized_demand'].values
    #
    # # Create the demand facility
    # # desalination_plant = Demand(demand_profile, nominal_power=1000)
    #
    # # Create a Battery system
    # battery = BatterySystem()
    #
    # nt = len(wind_speed_profile)
    # time = [datetime(2016, 1, 1) + timedelta(hours=h) for h in range(nt)]
    #
    # # Create a MicroGrid with the given devices
    # # Divide the prices by thousand because they represent €/MWh and we need €/kWh
    # micro_grid = MicroGrid(solar_farm=solar_farm,
    #                        wind_farm=wind_farm,
    #                        battery_system=battery,
    #                        demand_system=desalination_plant,
    #                        time_arr=time,
    #                        LCOE_years=20,
    #                        spot_price=prices[:, 0] / 1000,
    #                        band_price=prices[:, 1] / 1000,
    #                        maxeval=100)
    # micro_grid.optimize()
    # res = micro_grid(micro_grid.solution, verbose=True)
    # micro_grid.x_fx.to_excel('results.xlsx')
    # # print(micro_grid.x_fx)
    #
    # micro_grid.plot()
    # micro_grid.plot_optimization()
    #
    # battery.plot()
    #
    #
    #
    # plt.show()

    # print(res)


