
from Engine.CalculationEngine import GeneralDevice, Deposit, DataProfiles, SolarFarm, WindFarm, BatterySystem
from numpy import zeros, array, where, floor, c_, arange
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame, concat
import pandas as pd
import multiprocessing
from PyQt5.QtCore import QThread, QRunnable, pyqtSignal
from pySOT import *
from poap.controller import ThreadController, BasicWorkerThread, SerialController
plt.style.use('ggplot')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class OptimizationType:
    LCOE = 0,
    COST = 1,
    INCOME = 2,
    INCOME_COST_RATIO = 3,
    BENEFIT=4,
    ALL = 5


class SimplePump(GeneralDevice):

    def __init__(self, units, nominal_power, nominal_flow, unitary_cost, maintenance_perc_cost,
                 life, after_life_investment):
        """
        Pump object
        :param units:
        :param nominal_power: Nominal power in kW
        :param nominal_flow: Nominal flow in m3/h
        :param unitary_cost:
        :param maintenance_perc_cost:
        :param life:
        :param after_life_investment:
        """
        GeneralDevice.__init__(self, unitary_cost*units, maintenance_perc_cost, life, after_life_investment)

        self.units = units

        self.nominal_power = nominal_power

        self.nominal_flow = nominal_flow

        self.power_arr = None

        self.performance = None

    def power(self, Q):
        """
        Returns the power consumed by the pump(s)
        :param Q: Flow (m3/s)
        :return: The pump(s) power in kW
        """

        idx = where(Q > 0)[0]
        n = len(Q)

        p_max = self.units * self.nominal_power

        # energy in kilo-Watts-hour (kWh)
        self.power_arr = zeros(n)
        self.power_arr[idx] = Q[idx] * self.nominal_power / self.nominal_flow

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

        investment = self.investment_cost

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


class SimpleReverseOsmosisSystem(GeneralDevice):

    def __init__(self, units, aux_power, nominal_unit_flow_in, nominal_unit_flow_out, unitary_cost, maintenance_perc_cost,
                 life, after_life_investment):

        """

        :param units: Number of units
        :param aux_power: Power of the auxiliaries
        :param nominal_unit_flow_in: Nominal water flow in (sea water)
        :param nominal_unit_flow_out: Nominal water flow out (desalinated)
        :param unitary_cost:
        :param maintenance_perc_cost:
        :param life:
        :param after_life_investment:
        """
        GeneralDevice.__init__(self, unitary_cost * units, maintenance_perc_cost, life, after_life_investment)

        self.units = units

        self.aux_power = aux_power

        self.nominal_unit_flow_in = nominal_unit_flow_in

        self.nominal_unit_flow_out = nominal_unit_flow_out

    def plot(self, title='Reverse osmosys system'):
        print()

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
        inc = int(self.life+1)
        for idx in range(inc, N, inc):
            arr[idx] += investment * self.after_life_investment  # incur percentage of the investment after life

        return arr


class SimpleDesalinationPlant:

    def __init__(self, pumps: SimplePump, deposit: Deposit, reverse_osmosis_system: SimpleReverseOsmosisSystem):

        self.pumps = pumps

        self.deposit = deposit

        self.reverse_osmosis_system = reverse_osmosis_system

        self.minimum_power = self.reverse_osmosis_system.aux_power + self.pumps.nominal_power

        self.results_df = None

    def process_power(self, power_in, time):
        """
        Process an amount of power sent
        :param power_in: array of power values
        :param time: time matching those values in hourly steps
        :return:
        """

        # compute the minimum unitary power that can be processed
        unit_power = self.reverse_osmosis_system.aux_power + self.pumps.nominal_power

        # desalination power per unit of reverse osmosis
        dpu = power_in / unit_power

        # truncate the power to the power effectively usable by the plant
        desalination_units_used = floor(dpu)

        idx = where(desalination_units_used > self.reverse_osmosis_system.units)[0]
        desalination_units_used[idx] = self.reverse_osmosis_system.units

        # the power that can be processed without looking at the deposit is:
        power_taken_1 = desalination_units_used * unit_power

        # water produced by using the previously computed power
        r = self.reverse_osmosis_system.nominal_unit_flow_out / unit_power
        water_produced_1 = power_taken_1 * r

        # "send" the water request to the deposit and see if it fits.
        # the deposit returns the water that cannot be taken
        # the water demand is automatically taken into account
        water_mismatch = self.deposit.process_water(water_produced_1, time)

        # convert the water mismatch to power mismatch
        power_mismatch = water_mismatch / r

        # the final power that was possible to take is the power send, minus the positive power mismatch
        idx2 = where(power_mismatch > 0)[0]
        power_used = power_taken_1
        power_used[idx2] = power_taken_1[idx2] - power_mismatch[idx2]

        water_produced = power_used * r + water_mismatch

        k = self.reverse_osmosis_system.nominal_unit_flow_in / self.reverse_osmosis_system.nominal_unit_flow_out
        water_intake = water_produced * k

        power_not_used = power_in - power_used

        # results DataFrame
        n = len(power_in)
        data = c_[self.deposit.water[:n],
                  self.deposit.water_demand[:n],
                  self.deposit.water_flow[:n],
                  water_produced_1,
                  water_mismatch,
                  water_produced,
                  water_intake,
                  power_in,
                  power_taken_1,
                  power_mismatch,
                  power_used,
                  power_not_used]
        cols = ['water',
                'water demand',
                'water flow',
                'water sent',
                'water mismatch',
                'water actually produced',
                'water taken from the sea',
                'power in (available)',
                'power proposed',
                'power mismatch',
                'power actually used',
                'power not used']
        self.results_df = DataFrame(data=data, columns=cols, index=time[:n])

        return power_not_used, water_mismatch

    def plot(self):
        """

        :return:
        """
        self.reverse_osmosis_system.plot(title='Membranes')

        # self.pumps.plot(title='Pumps')

        self.deposit.plot(title='Storage Deposit')

    def get_cost(self, N):
        """
        Number of years
        :param N:
        :return:
        """

        arr = self.pumps.get_cost(N)
        arr += self.deposit.get_cost(N)
        arr += self.reverse_osmosis_system.get_cost(N)
        return arr


class Simulator(QThread):
    progress_signal = pyqtSignal(float)
    progress_text_signal = pyqtSignal(str)
    done_signal = pyqtSignal()

    def __init__(self, plant: SimpleDesalinationPlant, wind: WindFarm, solar: SolarFarm, battery: BatterySystem,
                 profiles: DataProfiles, max_eval=100, opt_type=OptimizationType.LCOE):
        """

        :param plant:
        :param wind:
        :param solar:
        :param battery:
        """

        QThread.__init__(self)

        if battery is None:
            self.dim = 2
            xup = [solar.max_power, wind.max_power]
            xlw = [solar.min_power, wind.min_power]
        else:
            self.dim = 3
            xup = [solar.max_power, wind.max_power, battery.max_energy]
            xlw = [solar.min_power, wind.min_power, battery.min_energy]

        # variables for the optimization
        self.xlow = array(xlw)  # lower bounds
        self.xup = array(xup)
        self.info = "Microgrid with Wind turbines, Photovoltaic panels and storage coupled to a demand"  # info
        self.integer = array([0])  # integer variables
        self.continuous = arange(1, self.dim)  # continuous variables

        self.max_eval = max_eval

        self.optimization_type = opt_type

        self.lcoe_years = 25

        self.investment_rate = 0.03

        self.plant = plant

        self.solar = solar

        self.wind = wind

        self.battery = battery

        self.profiles = profiles

        self.iteration = 0

        self.raw_results_df = list()

        self.results_df = None

        self.solution = None

        self.optimization_values = None

        self.cash_flow_df = DataFrame()

        self.X = list()

        self.Y = list()

    def set_state(self, x):
        """
        Set the size values
        :param x:
        :return:
        """
        self.solar.nominal_power = x[0]

        self.wind.nominal_power = x[1]

        if self.battery is not None:
            self.battery.nominal_energy = x[2]

    def land_usage(self, x):
        """
        Solar: https://www.qualenergia.it/sites/default/files/articolo-doc/Studio_NREL_FV_e_consumo_suolo.pdf
        (8.3 Acres / MW)

        Wind: https://www.nrel.gov/docs/fy09osti/45834.pdf
        (0.25 HA/MW)

        :param x:
        :return:
        """
        u = np.zeros(3)
        u[0] = 0.3358  # solar km2/MW
        u[1] = 0.0025  # wind km2/MW
        u[2] = 0.0  # storage

        # x comes in kW and kWh , we need it in MW and MWh
        return (x * 1e-3 * u).sum()

    def objfunction(self, x, verbose=False):
        """

        Args:
            x: optimsation vector [solar nominal power, wind nominal power, storage nominal power]

        Returns:

        """
        ################################################################################################################
        # Set the devices nominal power
        ################################################################################################################

        self.set_state(x)

        f, f_all = self.simulate()

        # add values
        self.raw_results_df.append(np.r_[f, x])

        self.X.append(x)
        self.Y.append(f_all)

        self.iteration += 1

        # progress
        p = self.iteration / self.max_eval * 100
        # print(p, '%')
        self.progress_signal.emit(p)

        return f

    def calculate_costs_array(self, N):
        """

        in the cost array we put in the item 0, the investment cost
        in the cells from 1 to N, we put:
        + the maintenance cost
        + the eventual replacement costs
        + the cost of purchasing electricity
        + the penalty of not supplying the water demand

        :return:
        """

        # there is a cost on the unsatisfied demand
        # when the water mismatch is negative it is due to an unsatisfied demand
        idx = np.where(self.plant.deposit.water_mismatch < 0)[0]
        water_penalty = (-self.plant.deposit.water_mismatch[idx] * self.profiles.water_price[idx]).sum()

        if self.battery is not None:
            idx = np.where(self.battery.grid_power > 0)[0]
            electricity_cost = (self.battery.grid_power[idx] * self.profiles.spot_price[idx] / 1000.0).sum()
        else:
            electricity_cost = 0

        # compute plant costs array
        y = np.arange(N + 1)
        arr = self.plant.get_cost(N)

        self.cash_flow_df['Desalination Plant cost'] = pd.Series(index=y, data=arr)

        if self.battery is not None:
            val = self.battery.get_cost(N)
            self.cash_flow_df['Battery cost'] = pd.Series(index=y, data=val)
            arr += val

        val = self.wind.get_cost(N)
        self.cash_flow_df['Wind cost'] = pd.Series(index=y, data=val)
        arr += val

        val = self.solar.get_cost(N)
        self.cash_flow_df['Solar cost'] = pd.Series(index=y, data=val)
        arr += val

        # add the water penalties and electricity costs
        arr[1:] += water_penalty + electricity_cost

        self.cash_flow_df['water penalty'] = pd.Series(index=y, data=np.r_[0, np.ones(N) * water_penalty])
        self.cash_flow_df['electricity cost'] = pd.Series(index=y, data=np.r_[0, np.ones(N) * electricity_cost])

        return arr

    def calculate_income_array(self, N):
        """
        in the income array we set the cell 0, to 0
        and we set the cells from 1 to N with:
        + income by sales of energy
        + income by sales of water
        :return:
        """

        # the water flow is negative for consumption
        w = self.results_df['water sent'].values - self.results_df['water flow'].values
        water_sales = (w * self.profiles.water_price).sum()

        y = np.arange(N + 1)
        self.cash_flow_df['Water sales'] = pd.Series(index=y, data=np.r_[0, np.ones(N)*water_sales])

        if self.battery is not None:
            neg = np.where(self.battery.grid_power < 0)[0]
            electricity_sales = (-self.battery.grid_power[neg] * self.profiles.spot_price[neg] / 1000.0).sum()
        else:
            electricity_sales = (self.results_df['power not used'].values * self.profiles.spot_price / 1000.0).sum()

        self.cash_flow_df['Electricity sales'] = pd.Series(index=y, data=np.r_[0, np.ones(N)*electricity_sales])

        total = water_sales + electricity_sales

        arr = np.zeros(N + 1)

        arr[1:] = total

        return arr

    def lcoe_calc(self, N, r):
        """
        Compute the LCOE value in €/kWh

        N: number of years considered i.e. 20
        r: discount rate i.e. 0.05

        d = [(1+r)^n for n in range(N)]

        a = sum(costs_array / d)

        b = sum(income_array / d)

        lcoe = a / b
        :param r: discount rate
        :return:
        """

        costs_array = self.calculate_costs_array(N)

        income_array = self.calculate_income_array(N)

        p = self.results_df['power not used'].values.sum()

        p_array = np.array([p for n in range(N+1)])

        d = np.array([(1 + r)**n for n in range(N+1)])

        a = sum(costs_array / d)

        b = sum(p_array / d)

        lcoe = a / b

        return lcoe, income_array, costs_array

    def simulate(self, N=None):
        """

        :param N: number of items to simulate from  the profiles
        :return:
        """

        if N is None or N > len(self.profiles.time):
            N = len(self.profiles.time)

        # Compute the raw production
        p_res = (self.solar.power() + self.wind.power())[:N]

        if self.battery is not None:
            # If there is a electrochemical battery:

            # if the plant start-up power is not supplied, request that power t the battery
            desired_p_on_battery = self.plant.minimum_power * self.plant.reverse_osmosis_system.units

            # desired_p_on_battery = self.plant.minimum_power
            idx = where(p_res < desired_p_on_battery)[0]
            req_p_battery = zeros(N)
            req_p_battery[idx] = self.plant.minimum_power - p_res[idx]
            p_res2 = p_res + req_p_battery
        else:
            # if there is no electrochemical battery:
            req_p_battery = zeros(N)
            p_res2 = p_res

        # compute the desalination
        power_surplus_, water_mismatch_ = self.plant.process_power(p_res2, self.profiles.time[:N])

        if self.battery is not None:
            # include the surplus as charging power for the battery
            req_p_battery -= power_surplus_

            # send power surplus to the battery storage
            bat_energy, bat_power, bat_grid_power, bat_soc = self.battery.simulate_array(P=req_p_battery,
                                                                                         soc_0=0.5,
                                                                                         time=self.profiles.time[:N])

            data = c_[p_res, req_p_battery, bat_energy, bat_power, bat_grid_power, bat_soc]
            cols = ['power from RES', 'power sent to the battery', 'battery energy',
                    'battery power', 'battery grid power', 'battery SoC']
            df = DataFrame(data=data, columns=cols, index=self.profiles.time[:N])

            # extend the plant results with the storage device results
            self.results_df = concat([self.plant.results_df, df], axis=1)

            # water mismatch:
            #   positive -> too much water (full)
            #   negative -> too few water (empty)
            w_mismatch = abs(self.results_df['water mismatch'].values.sum())

        else:

            # set the system results as the plants results as there is no battery
            self.results_df = self.plant.results_df

        # compute the objective
        lcoe, income, cost = self.lcoe_calc(self.lcoe_years, self.investment_rate)

        f_lcoe = abs(lcoe)
        f_cost = cost.sum()
        f_income = - income.sum()
        f_income_cost = - income.sum() / cost.sum()
        f_benefit = cost.sum() - income.sum()

        # return
        if self.optimization_type is OptimizationType.LCOE:
            f = abs(lcoe)
        elif self.optimization_type is OptimizationType.COST:
            f = cost.sum()
        elif self.optimization_type is OptimizationType.INCOME:
            f = - income.sum()
        elif self.optimization_type is OptimizationType.INCOME_COST_RATIO:
            f = - income.sum() / cost.sum()
        elif self.optimization_type is OptimizationType.BENEFIT:
            f = (cost.sum() - income.sum())  # -(income.sum() / cost.sum())

        f_all = [f_lcoe, f_cost, f_income, f_income_cost, f_benefit]

        return f, f_all

    def run(self):
        """
        Function that optimizes a MicroGrid Object
        Args:

        Returns:

        """
        self.iteration = 0
        self.raw_results_df = list()
        self.X = list()
        self.Y = list()

        self.progress_signal.emit(0)
        self.progress_text_signal.emit('Optimizing facility sizes by surrogate optimization: Objective: ' + str(self.optimization_type))

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
        # controller = SerialController(self.objfunction)
        controller = ThreadController()

        # (5) Use the synchronous strategy without non-bound constraints
        strategy = SyncStrategyNoConstraints(
            worker_id=0, data=self, maxeval=self.max_eval, nsamples=1,
            exp_design=exp_des, response_surface=surrogate,
            sampling_method=adapt_samp)
        controller.strategy = strategy

        # Launch the threads and give them access to the objective function
        # for _ in range(multiprocessing.cpu_count()):
        #     worker = BasicWorkerThread(controller, self.objfunction)
        #     controller.launch_worker(worker)

        worker = BasicWorkerThread(controller, self.objfunction)
        controller.launch_worker(worker)

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

        # format the trials DataFrame
        data = np.array(self.raw_results_df)

        if self.battery is not None:
            cols = ['solar (kW)', 'wind (kW)', 'storage (kWh)']
        else:
            cols = ['solar (kW)', 'wind (kW)']
        self.raw_results_df = pd.DataFrame(data=data[:, 1:], columns=cols, index=data[:, 0])
        self.raw_results_df.sort_index(inplace=True)

        self.progress_text_signal.emit('Done!')
        self.done_signal.emit()

    def run_state_at(self, idx):
        """
        Set the optimization values of the option idx to the current simulator state
        :param idx: index
        :return:
        """

        x = self.raw_results_df.iloc[idx]

        self.set_state(x)

        self.simulate()

    def plot(self, ax=None):
        """
        Plot the optimization convergence
        Returns:

        """
        clr = np.array(['#2200CC', '#D9007E', '#FF6600', '#FFCC00', '#ACE600', '#0099CC',
                        '#8900CC', '#FF0000', '#FF9900', '#FFFF00', '#00CC01', '#0055CC'])
        if self.optimization_values is not None:
            max_eval = len(self.optimization_values)

            if ax is None:
                f, ax = plt.subplots()
            # Points
            ax.scatter(np.arange(0, max_eval), self.optimization_values, color=clr[6])
            # Best value found
            ax.plot(np.arange(0, max_eval), np.minimum.accumulate(self.optimization_values), color=clr[1], linewidth=3.0)
            ax.set_xlabel('Evaluations')
            ax.set_ylabel('Function Value')
            ax.set_title('Optimization convergence')

    def plot_clusters(self, ax=None, n=5):
        """
        Performs the cluster analysis of the results
        :param n:
        :return:
        """
        from sklearn.cluster import KMeans
        x = self.raw_results_df.index.values.reshape(-1, 1)
        res = KMeans(n_clusters=n, random_state=0).fit(x)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        print('\n\nClusters:')

        clr = np.array(['#2200CC', '#D9007E', '#FF6600', '#FFCC00', '#ACE600', '#0099CC',
                        '#8900CC', '#FF0000', '#FF9900', '#FFFF00', '#00CC01', '#0055CC'])

        y = self.raw_results_df.values.sum(axis=1)
        ax.scatter(x, y, color=clr[res.labels_.astype(int)])
        ax.set_xlabel('Objective')
        ax.set_ylabel('Installed power')
        ax.set_title('Results clustering')

        print('\n\n')

    def plot_3D(self, fig, x_serie='solar (kW)', y_serie='wind (kW)', z_label='', surface=False):
        """
        :param x_serie:
        :param y_serie:
        :param z_serie:
        :return:
        """
        # Make data.
        X = self.raw_results_df[x_serie].values
        Y = self.raw_results_df[y_serie].values
        Z = self.raw_results_df.index.values.reshape(-1, 1)

        # Scatter plot
        # fig = plt.figure(figsize=(16, 8))
        ax = fig.gca(projection='3d')

        if surface:
            ax.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True)
        else:
            ax.scatter(X, Y, Z)

        ax.set_xlabel(x_serie)
        ax.set_ylabel(y_serie)
        ax.set_zlabel(z_label)
        ax.set_facecolor('white')


    def economic_sensitivity(self, years_arr, inv_rate_arr):
        """

        :param years_arr:
        :param inv_rate_arr:
        :return:
        """
        ny = len(years_arr)
        ni = len(inv_rate_arr)
        values = np.zeros((3, ni, ny))

        for i, years in enumerate(years_arr):
            for j, inv_rate in enumerate(inv_rate_arr):
                lcoe, income, costs = self.lcoe_calc(years, inv_rate)

                values[:, j, i] = np.array([lcoe, income.sum(), costs.sum()])

        df_lcoe = pd.DataFrame(data=values[0, :, :], index=inv_rate_arr, columns=years_arr)

        return df_lcoe

    def text_report(self, obj_fun_dict):

        inv_map = {v: k for k, v in obj_fun_dict.items()}

        val = 'Conditions: \n\n'

        val += 'Amortization years:\t' + '{0:.0f}'.format(self.lcoe_years) + '\n'
        val += 'Discount rate:\t' + '{0:.2f}'.format(self.investment_rate) + '\n'
        val += 'Optimizing for:\t' + inv_map[self.optimization_type] + '\n'

        val += '\n'

        val += 'Results: \n\n'

        if self.solution is not None:
            s = self.solution
        else:
            s = [0, 0, 0, 0]

        # val += 'Demand size:\t' + '{0:.2f}'.format(self.simulator.demand_system.nominal_power) + ' kW.\n'
        if self.solar is not None:
            val += 'Solar farm size:\t' + '{0:,.2f}'.format(self.solar.nominal_power) + \
                   '/ best:({0:,.2f})'.format(s[0]) + ' kW.\n'
        if self.wind is not None:
            val += 'Wind farm size:\t' + '{0:,.2f}'.format(self.wind.nominal_power) + \
                   '/ best:({0:,.2f})'.format(s[1]) + ' kW.\n'
        if self.battery is not None:
            val += 'Storage size:\t' + '{0:,.2f}'.format(self.battery.nominal_energy) + \
                   '/ best:({0:,.2f})'.format(s[2]) + ' kWh.\n'

        val += '\n'
        if self.plant is not None:
            val += 'Desal. plant cost:\t' + '{0:,.2f}'.format(self.plant.get_cost(1)[0]) + ' €.\n'
        if self.solar is not None:
            val += 'Solar farm cost:\t' + '{0:,.2f}'.format(self.solar.get_cost(1)[0]) + ' €.\n'
        if self.wind is not None:
            val += 'Wind farm cost:\t' + '{0:,.2f}'.format(self.wind.get_cost(1)[0]) + ' €.\n'
        if self.battery is not None:
            val += 'Storage cost:\t' + '{0:,.2f}'.format(self.battery.get_cost(1)[0]) + ' €.\n'

        val += '\n'

        lcoe_, income_array, costs_array = self.lcoe_calc(N=self.lcoe_years, r=self.investment_rate)
        benefit = income_array - costs_array

        val += 'Investment costs:\t' + '{0:,.2f}'.format(costs_array[0]) + ' €.\n'
        val += 'Total costs:\t\t' + '{0:,.2f}'.format(costs_array.sum()) + ' €.\n'
        val += 'Total income:\t' + '{0:,.2f}'.format(income_array.sum()) + ' €.\n'
        val += 'Average benefit:\t' + '{0:,.2f}'.format(benefit.mean()) + ' €/year.\n'

        val += 'LCOE:\t\t' + '{0:.6f}'.format(lcoe_) + ' €/kWh.\n'

        return val

    def export(self, file_name):
        """
        Save results to excel file
        :param file_name:  file name ending with .xlsx
        :return:
        """

        # make the report as DF

        lcoe_, income_array, costs_array = self.lcoe_calc(N=self.lcoe_years, r=self.investment_rate)
        benefit = income_array - costs_array

        d = [['Amortization years:', self.lcoe_years, ''],
             ['Amortization years:', self.investment_rate, ''],
             ['', np.NaN, ''],
             ['Solar farm size:', self.solar.nominal_power, 'kW'],
             ['Wind farm size:', self.wind.nominal_power, 'kW'],
             ['Battery farm size:', self.battery.nominal_energy, 'kW'],
             ['', np.NaN, ''],
             ['Solar investment:', self.solar.get_cost(1)[0], '€'],
             ['Wind investment:', self.wind.get_cost(1)[0], '€'],
             ['Battery investment:', self.battery.get_cost(1)[0], '€'],
             ['', np.NaN, ''],
             ['Total investment:', costs_array[0], '€'],
             ['Total cost:', costs_array.sum(), '€'],
             ['Total income:', income_array.sum(), '€'],
             ['Average benefit:', benefit.mean(), '€/year'],
             ['LCOE:', lcoe_, '€/kWh'],
             ]

        rep_df = pd.DataFrame(data=d, columns=['magnitude', 'value', 'units'])
        writer = pd.ExcelWriter(file_name)
        rep_df.to_excel(writer, 'Report', index=False)

        if self.cash_flow_df is not None:
            self.cash_flow_df.to_excel(writer, 'Cash_flow')
        if self.raw_results_df is not None:
            self.raw_results_df.to_excel(writer, 'Raw results')
        if self.results_df is not None:
            self.results_df.to_excel(writer, 'Results')

        writer.save()


if __name__ == '__main__':

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
                           maintenance_perc_cost=20,
                           life=25,
                           after_life_investment=0.8)
    solar_farm.nominal_power = 5000

    wind_farm = WindFarm(profile=profiles.wind_speed,
                         wt_curve_df=profiles.wind_turbine_curve,
                         wind_power_min=100,
                         wind_power_max=10000,
                         unitary_cost=900,
                         maintenance_perc_cost=30,
                         life=25,
                         after_life_investment=0.8)
    wind_farm.nominal_power = 1000

    battery = BatterySystem(charge_efficiency=0.9, discharge_efficiency=0.9, max_soc=0.99, min_soc=0.3,
                            battery_energy_min=0, battery_energy_max=100000, unitary_cost=900,
                            maintenance_perc_cost=0.2, life=7, after_life_investment=1)
    battery.nominal_energy = 10000

    plant.deposit.SoC_0 = 0.00000001
    sim = Simulator(plant=plant, solar=solar_farm, wind=wind_farm, battery=battery, profiles=profiles)
    # sim = Simulator(plant=plant, solar=solar_farm, wind=wind_farm, battery=None, profiles=profiles)
    f = sim.simulate()

    print(sim.results_df)
    print(f)
    sim.results_df.to_excel('simple_res.xlsx')

    # optimize
    sim.max_eval = 50
    sim.optimization_type = OptimizationType.INCOME_COST_RATIO
    sim.run()
    print('sol: ', sim.solution)
    print(sim.raw_results_df.head(20))
    sim.results_df.to_excel('simple_res_opt.xlsx')
    sim.raw_results_df.to_excel('raw_res.xlsx')
    # plant.plot()
    # sim.plot()
    sim.plot_clusters(n=5)

    print('lcoe:', sim.lcoe_calc(20, 0.03), '€/kWh')

    plt.show()
