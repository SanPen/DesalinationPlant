# This file is part of GridCal.
#
# GridCal is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GridCal is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GridCal.  If not, see <http://www.gnu.org/licenses/>.

import os.path
from datetime import datetime, timedelta
import sys
from collections import OrderedDict
from enum import Enum
from Gui.gui import *
from PyQt5.QtWidgets import *
from Gui.GuiFunctions import PandasModel
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import cpu_count

from Engine.CalculationEngine import *
from Engine.SimpleCalculationEngine import *

__author__ = 'Santiago Peñate Vera'
__VERSION__ = "1.2"

"""
This class is the handler of the main gui of GridCal.
"""

########################################################################################################################
# Main Window
########################################################################################################################


class MainGUI(QMainWindow):
    # Prices
    profiles = None

    #
    project_directory = None

    simulator = None

    def __init__(self, parent=None):
        """

        @param parent:
        """

        # create main window
        QWidget.__init__(self, parent)
        self.ui = Ui_mainWindow()
        self.ui.setupUi(self)

        self.available_results = list()
        self.available_results.append('Optimization plot')
        self.available_results.append('Cluster plot')

        self.available_results.append('3D Solar-Wind')
        self.available_results.append('3D Solar-Storage')
        self.available_results.append('3D Wind-Storage')

        re = [  'water',
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
                'power not used',
                'power from RES',
                'power sent to the battery',
                'battery energy',
                'battery power',
                'battery grid power',
                'battery SoC']
        for r in re:
            self.available_results.append(r)
        self.ui.results_comboBox.addItems(self.available_results)

        self.obj_fun_dict = dict()

        '''
        LCOE = 0,
        COST = 1,
        INCOME = 2,
        INCOME_COST_RATIO = 3
        '''
        self.obj_fun_dict['0_BENEFIT'] = OptimizationType.BENEFIT
        self.obj_fun_dict['1_MIN_LCOE'] = OptimizationType.LCOE
        self.obj_fun_dict['2_MIN_COST'] = OptimizationType.COST
        self.obj_fun_dict['3_MAX_INCOME'] = OptimizationType.INCOME
        self.obj_fun_dict['4_MAX_INCOME_COST_RATIO'] = OptimizationType.INCOME_COST_RATIO
        k = list(self.obj_fun_dict.keys())
        k.sort()
        self.ui.obj_function_comboBox.addItems(k)


        self.ui.info_label.setText(__author__ + ', Desalination plant v' + __VERSION__)

        ################################################################################################################
        # Connections
        ################################################################################################################
        self.ui.actionNew_project.triggered.connect(self.new_project)

        self.ui.actionOpen_file.triggered.connect(self.open_file)

        self.ui.actionSave.triggered.connect(self.save_results)

        # # Buttons

        self.ui.deposit_demand_plot_pushButton.clicked.connect(self.plot_water_demand)

        self.ui.deposit_water_price_pushButton.clicked.connect(self.plot_water_price)

        self.ui.spot_price_pushButton.clicked.connect(self.plot_spot_price)

        self.ui.secondary_price_plot_pushButton.clicked.connect(self.plot_secondary_price)

        self.ui.wind_speed_plot_pushButton.clicked.connect(self.plot_wind)

        self.ui.solar_plot_irradiation_pushButton.clicked.connect(self.plot_solar)

        self.ui.wind_turbine_curve_plot_pushButton.clicked.connect(self.plot_ag_curve)

        self.ui.analyze_selected_pushButton.clicked.connect(self.analyze_selected_result)

        self.ui.plot_results_pushButton.clicked.connect(self.plot_results)

        # run button
        self.ui.size_simulate_pushButton.clicked.connect(self.run_sizing_simulation)

        ################################################################################################################
        # Connections
        ################################################################################################################

        self.unlock()

        self.set_default_ui_values()

        #     load data (or try..)
        try:
            self.profiles = DataProfiles('../Engine/data_mod.xls')
        except:
            pass

    def set_ui_state(self, val=True):
        self.ui.progress_frame.setVisible(val)

    def lock(self):
        self.set_ui_state(True)

    def unlock(self):
        self.set_ui_state(False)

    def set_default_ui_values(self):
        """
        Set the UI default values
        :return:
        """
        # Desalination plant : Membranes
        self.ui.membrane_number_spinBox.setValue(6)
        self.ui.membrane_aux_power_doubleSpinBox.setValue(41.16),  # 59/3
        self.ui.membrane_unit_flow_in_doubleSpinBox.setValue(285),  # 280.16/3
        self.ui.membrane_unit_flow_out_doubleSpinBox.setValue(180),  # 100.2/3
        self.ui.desalination_building_after_live_investment_doubleSpinBox.setValue(1.0)
        self.ui.desalination_building_maintenance_perc_doubleSpinBox.setValue(0.05)  # 0.18
        self.ui.desalination_building_unitary_cost_doubleSpinBox.setValue(1.0)
        self.ui.desalination_building_life_spinBox.setValue(10)  # 8

        # Desalination plant: Deposit
        self.ui.deposit_maintenance_perc_doubleSpinBox.setValue(0.01)
        self.ui.deposit_after_life_investment_doubleSpinBox.setValue(0.2)
        self.ui.deposit_life_spinBox.setValue(50)
        self.ui.deposito_investment_cost_doubleSpinBox.setValue(450000)
        self.ui.deposit_capacity_doubleSpinBox.setValue(180000)
        self.ui.deposit_head_doubleSpinBox.setValue(190)

        # Desalination plant: sea pump
        self.ui.sea_pump_units_spinBox.setValue(8)  # 6
        self.ui.sea_pump_nominal_power_doubleSpinBox.setValue(110)  # 423/3
        self.ui.sea_pump_nominal_flow_doubleSpinBox.setValue(115)  # 285/3
        self.ui.sea_pump_after_life_investent_doubleSpinBox.setValue(1.0)
        self.ui.sea_pump_life_spinBox.setValue(25)
        self.ui.sea_pump_unitary_cost_doubleSpinBox.setValue(1110)
        self.ui.sea_pump_maintenance_cost_doubleSpinBox.setValue(0.05)

        # Solar plant
        self.ui.solar_enable_checkBox.setChecked(True)
        self.ui.solar_unitary_cost_doubleSpinBox.setValue(500)  # 700
        self.ui.solar_panel_eff_doubleSpinBox.setValue(0.20)
        self.ui.solar_maintenance_perc_cost_doubleSpinBox.setValue(0.05)  # 0.2
        self.ui.solar_panel_size_doubleSpinBox.setValue(1.6)
        self.ui.solar_life_span_spinBox.setValue(25)
        self.ui.solar_min_power_doubleSpinBox.setValue(1)
        self.ui.solar_max_power_doubleSpinBox.setValue(10000)
        self.ui.solar_land_usage_rate_doubleSpinBox.setValue(3.358e-4)
        self.ui.solar_land_cost_doubleSpinBox.setValue(100000)

        # Wind plant
        self.ui.wind_enable_checkBox.setChecked(True)
        self.ui.wind_unitary_cost_doubleSpinBox.setValue(850)  # 900
        self.ui.wind_maintenance_cost_doubleSpinBox.setValue(0.05)  # 0.3
        self.ui.wind_after_life_investment_doubleSpinBox.setValue(0.8)
        self.ui.wind_min_power_doubleSpinBox.setValue(1)
        self.ui.wind_max_power_doubleSpinBox.setValue(10000)
        self.ui.wind_land_usage_rate_doubleSpinBox.setValue(2.5e-6)
        self.ui.wind_land_cost_doubleSpinBox.setValue(100000)

        # Storage plant
        self.ui.storage_enable_checkBox.setChecked(True)
        self.ui.storage_charge_eff_doubleSpinBox.setValue(0.9)
        self.ui.storage_discharge_eff_doubleSpinBox.setValue(0.9)
        self.ui.storage_max_soc_doubleSpinBox.setValue(0.99)
        self.ui.storage_min_soc_doubleSpinBox.setValue(0.22)
        self.ui.storage_unitary_cost_doubleSpinBox.setValue(400)  # 900
        self.ui.storage_maintenance_cost_doubleSpinBox.setValue(0.05)  # 0.2
        self.ui.storage_life_spinBox.setValue(8)
        self.ui.storage_after_life_investment_doubleSpinBox.setValue(0.8)
        self.ui.storage_min_power_doubleSpinBox.setValue(0.8)
        self.ui.storage_max_power_doubleSpinBox.setValue(10000)

        # Grid connection
        self.ui.grid_enable_checkBox.setChecked(True)
        self.ui.grid_connection_power_doubleSpinBox.setValue(0)

    def make_simulation_object(self):
        """

        :return:
        """

        # assert(self.demand_profile is not None)
        if self.profiles is None:
            self.msg('There are no profiles')
            return

        # Desalination plant : Membranes ###############################################################################
        osmosis = SimpleReverseOsmosisSystem(units=self.ui.membrane_number_spinBox.value(),
                                             aux_power=self.ui.membrane_aux_power_doubleSpinBox.value(),
                                             nominal_unit_flow_in=self.ui.membrane_unit_flow_in_doubleSpinBox.value(),
                                             nominal_unit_flow_out=self.ui.membrane_unit_flow_out_doubleSpinBox.value(),
                                             unitary_cost=self.ui.desalination_building_unitary_cost_doubleSpinBox.value(),
                                             maintenance_perc_cost=self.ui.desalination_building_maintenance_perc_doubleSpinBox.value(),
                                             life=self.ui.desalination_building_life_spinBox.value(),
                                             after_life_investment=self.ui.desalination_building_after_live_investment_doubleSpinBox.value())

        # Desalination plant: Deposit
        deposit = Deposit(capacity=self.ui.deposit_capacity_doubleSpinBox.value(),
                          head=self.ui.deposit_head_doubleSpinBox.value(),
                          water_demand=self.profiles.water_demand,
                          investment_cost=self.ui.deposito_investment_cost_doubleSpinBox.value(),
                          maintenance_perc_cost=self.ui.deposit_maintenance_perc_doubleSpinBox.value(),
                          life=self.ui.deposit_life_spinBox.value(),
                          after_life_investment=self.ui.deposit_after_life_investment_doubleSpinBox.value())

        # Desalination plant: sea pump
        pumps = SimplePump(units=self.ui.sea_pump_units_spinBox.value(),
                           nominal_power=self.ui.sea_pump_nominal_power_doubleSpinBox.value(),
                           nominal_flow=self.ui.sea_pump_nominal_flow_doubleSpinBox.value(),
                           unitary_cost=self.ui.sea_pump_unitary_cost_doubleSpinBox.value(),
                           maintenance_perc_cost=self.ui.sea_pump_maintenance_cost_doubleSpinBox.value(),
                           life=self.ui.sea_pump_life_spinBox.value(),
                           after_life_investment=self.ui.sea_pump_after_life_investent_doubleSpinBox.value())

        # complete desalination plant
        plant = SimpleDesalinationPlant(pumps=pumps, deposit=deposit, reverse_osmosis_system=osmosis)

        # Solar plant ##################################################################################################

        # self.ui.solar_plot_irradiation_pushButton
        self.ui.solar_panel_eff_doubleSpinBox.value()

        self.ui.solar_panel_size_doubleSpinBox.value()

        if self.ui.solar_enable_checkBox.isChecked():
            solar_farm = SolarFarm(self.profiles.solar_irradiation,
                                   solar_power_min=self.ui.solar_min_power_doubleSpinBox.value(),
                                   solar_power_max=self.ui.solar_max_power_doubleSpinBox.value(),
                                   unitary_cost=self.ui.solar_unitary_cost_doubleSpinBox.value(),
                                   maintenance_perc_cost=self.ui.solar_maintenance_perc_cost_doubleSpinBox.value(),
                                   life=self.ui.solar_life_span_spinBox.value(),
                                   after_life_investment=self.ui.solar_after_life_investment_doubleSpinBox.value())
        else:
            solar_farm = None

        # Wind plant ##################################################################################################

        if self.ui.wind_enable_checkBox.isChecked():
            wind_farm = WindFarm(self.profiles.wind_speed, self.profiles.wind_turbine_curve,
                                 wind_power_min=self.ui.wind_min_power_doubleSpinBox.value(),
                                 wind_power_max=self.ui.wind_max_power_doubleSpinBox.value(),
                                 unitary_cost=self.ui.wind_unitary_cost_doubleSpinBox.value(),
                                 maintenance_perc_cost=self.ui.wind_maintenance_cost_doubleSpinBox.value(),
                                 life=self.ui.wind_life_spinBox.value(),
                                 after_life_investment=self.ui.wind_after_life_investment_doubleSpinBox.value())
        else:
            wind_farm = None

        # Storage plant ################################################################################################
        if self.ui.storage_enable_checkBox.isChecked():
            battery = BatterySystem(charge_efficiency=self.ui.storage_charge_eff_doubleSpinBox.value(),
                                    discharge_efficiency=self.ui.storage_discharge_eff_doubleSpinBox.value(),
                                    max_soc=self.ui.storage_max_soc_doubleSpinBox.value(),
                                    min_soc=self.ui.storage_min_soc_doubleSpinBox.value(),
                                    battery_energy_min=self.ui.storage_min_power_doubleSpinBox.value(),
                                    battery_energy_max=self.ui.storage_max_power_doubleSpinBox.value(),
                                    unitary_cost=self.ui.storage_unitary_cost_doubleSpinBox.value(),
                                    maintenance_perc_cost=self.ui.storage_maintenance_cost_doubleSpinBox.value(),
                                    life=self.ui.storage_life_spinBox.value(),
                                    after_life_investment=self.ui.storage_after_life_investment_doubleSpinBox.value())
        else:
            battery = None

        # Grid connection ##############################################################################################
        if self.ui.grid_enable_checkBox.isChecked():

            grid = Grid(connection_power=self.ui.grid_connection_power_doubleSpinBox.value(),
                        spot_price=self.profiles.spot_price,
                        secondary_price=self.profiles.secondary_reserve_price,
                        unitary_cost=0,
                        maintenance_perc_cost=0,
                        life=10000000000,
                        after_life_investment=0)
        else:
            grid = None

        # Time profile #################################################################################################
        # nt = len(self.profiles.time)
        # start = self.ui.start_dateEdit.dateTime().toPyDateTime()
        # self.time = [start + timedelta(hours=h) for h in range(nt)]

        # simulation type ##############################################################################################
        sel = self.ui.obj_function_comboBox.currentText()
        obj_fun_type = self.obj_fun_dict[sel]

        # simulation object ############################################################################################
        self.simulator = Simulator(plant=plant,
                                   solar=solar_farm,
                                   wind=wind_farm,
                                   battery=battery,
                                   profiles=self.profiles,
                                   max_eval=self.ui.max_eval_spinBox.value(),
                                   opt_type=obj_fun_type)

    def new_project(self):
        print('new_project')

    def open_file(self):
        """

        :return:
        """

        # declare the allowed file types
        files_types = "Excel 97 (*.xls);;Excel (*.xlsx)"
        # call dialog to select the file

        filename, type_selected = QFileDialog.getOpenFileName(self, 'Open file',
                                                              directory=self.project_directory,
                                                              filter=files_types)

        if len(filename) > 0:
            # load file
            xl = pd.ExcelFile(filename)

            self.project_directory = os.path.dirname(filename)

            # assert that the requires sheets exist. This sort of determines if the excel file is the right one
            c1 = 'profiles' in xl.sheet_names
            c2 = 'AG_CAT' in xl.sheet_names

            cond = c1 and c2

            if cond:

                # store the working directory
                self.profiles = DataProfiles(filename)

                self.plot_spot_price()
            else:

                self.msg('The file format is not right.')

    def save_results(self):
        """

        :return:
        """
        if self.simulator is not None:
            files_types = "Excel (*.xlsx)"

            filename, type_selected = QFileDialog.getSaveFileName(self, 'Export results',
                                                                  directory=self.project_directory,
                                                                  filter=files_types)

            if len(filename) > 0:

                if not filename.endswith('.xlsx'):
                    filename += '.xlsx'

                if self.simulator is not None:
                    self.simulator.export(file_name=filename)

    def plot_text_results(self):
        """
        Print the test results
        :return:
        """

        self.ui.plainTextEdit.clear()

        val = self.simulator.text_report(self.obj_fun_dict)

        self.ui.plainTextEdit.setPlainText(val)

    def plot_results(self):
        """
        Plot the simulation results
        :return:
        """

        '''
        results_df

        water
        water demand
        water flow
        water sent
        water mismatch
        water actually produced
        water taken from the sea
        power in (available)
        power proposed
        power mismatch
        power actually used
        power not used
        power from RES
        power sent to the battery
        battery energy
        battery power
        battery grid power
        battery SoC

        '''

        if self.simulator is not None:

            sel = self.ui.results_comboBox.currentText()
            self.ui.resultsPlot.clear(force=True)
            ax = self.ui.resultsPlot.get_axis()
            fig = self.ui.resultsPlot.get_figure()

            if sel == 'Optimization plot':

                self.simulator.plot(ax=ax)

            elif sel == 'Cluster plot':
                self.simulator.plot_clusters(ax=ax, n=10)

            elif sel == '3D Solar-Wind':
                self.simulator.plot_3D(fig, x_serie='solar (kW)', y_serie='wind (kW)', z_label='Objetive', surface=False)

            elif sel == '3D Solar-Storage':
                self.simulator.plot_3D(fig, x_serie='solar (kW)', y_serie='storage (kWh)', z_label='Objetive', surface=False)

            elif sel == '3D Wind-Storage':
                self.simulator.plot_3D(fig, x_serie='wind (kW)', y_serie='storage (kWh)', z_label='Objetive', surface=False)

            else:
                if sel in self.simulator.results_df.columns.values:
                    self.simulator.results_df[sel].plot(ax=ax)
                else:
                    self.msg(sel + ' is not in the results list')

            self.ui.resultsPlot.redraw()

    def plot_input(self, arr, ylabel='', xlabel='', title=''):
        """

        :param arr:
        :param ylabel:
        :param xlabel:
        :param title:
        :return:
        """
        self.ui.mainPlot.clear()
        ax = self.ui.mainPlot.get_axis()

        ax.plot(self.profiles.time, arr)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        self.ui.mainPlot.redraw()

    def plot_water_price(self):
        """

        :return:
        """
        if self.profiles is not None:
            self.plot_input(self.profiles.water_price, ylabel='€/m3', xlabel='Time', title='Water price')

    def plot_water_demand(self):
        """

        :return:
        """
        if self.profiles is not None:
            self.plot_input(self.profiles.water_demand, ylabel='m3', xlabel='Time', title='Water demand')

    def plot_spot_price(self):
        """

        :return:
        """
        if self.profiles is not None:
            self.plot_input(self.profiles.spot_price, ylabel='€/kWh', xlabel='Time', title='SPOT price')

    def plot_secondary_price(self):
        """

        :return:
        """
        if self.profiles is not None:
            self.plot_input(self.profiles.secondary_reserve_price, ylabel='€/kWh', xlabel='Time', title='Secondary reserve price')

    def plot_wind(self):
        """

        :return:
        """
        if self.profiles is not None:
            self.plot_input(self.profiles.wind_speed, ylabel='m/s', xlabel='Time', title='Wind speed')

    def plot_ag_curve(self):
        """

        :return:
        """
        if self.profiles is not None:
            self.ui.mainPlot.clear()
            ax = self.ui.mainPlot.get_axis()

            ax.plot(self.profiles.wind_turbine_curve)
            ax.set_xlabel('wind speed')
            ax.set_ylabel('kW')
            ax.set_title('Wind turbine curve')

            self.ui.mainPlot.redraw()

    def plot_solar(self):
        """

        :return:
        """
        if self.profiles is not None:
            self.plot_input(self.profiles.solar_irradiation, ylabel='W/m^2', xlabel='Time', title='Irradiation')

    def analyze_selected_result(self):
        print('analyze_selected_result')
        if self.simulator is not None:

            # get the simulated x
            idx = self.ui.results_tableView.selectedIndexes()[0].row()
            print(idx)

            self.simulator.run_state_at(idx)

            self.post_sizing_simulation()

        else:
            self.msg('There is no simulation in memory, run one ;)')

    def msg(self, text):
        """
        Message box
        :param text:
        :return:
        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text)
        # msg.setInformativeText("This is additional information")
        msg.setWindowTitle("Aviso")
        # msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()

    def run_sizing_simulation(self):
        """
        Run asynchronous simulation
        :return:
        """
        self.lock()

        self.make_simulation_object()

        self.simulator.progress_signal.connect(self.ui.progressBar.setValue)
        self.simulator.progress_text_signal.connect(self.ui.progress_label.setText)
        self.simulator.done_signal.connect(self.unlock)
        self.simulator.done_signal.connect(self.post_sizing_simulation)
        self.simulator.start()

    def post_sizing_simulation(self):
        """

        :return:
        """
        # set the solution as the current micro grid state
        # res = self.simulator(self.simulator.solution, verbose=True)

        print('Done!')

        # self.micro_grid.plot()
        # plt.plot()
        self.plot_text_results()
        self.plot_results()
        self.ui.results_tableView.setModel(PandasModel(self.simulator.raw_results_df, editable=True))

        years_arr = [10, 15, 20, 25, 30, 40]
        inv_rate_arr = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        df_lcoe = self.simulator.economic_sensitivity(years_arr, inv_rate_arr)
        self.ui.economic_tableView.setModel(PandasModel(df_lcoe))

        try:
            self.ui.cash_flow_tableView.setModel(PandasModel(self.simulator.cash_flow_df, editable=False))
        except:
            warn('Could not display the cash flow...try exporting.')


def run():
    app = QApplication(sys.argv)
    window = MainGUI()
    window.resize(1.61 * 700, 700)  # golden ratio
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run()
