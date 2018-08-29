from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd


def plot_2D(file_name, df_X, df_Y,  z_serie, n=10):

    if df_X.shape[0] > n:
        # y = df_X.values[:, 0:2].sum(axis=1)  # sum Solar + Wind
        # x = df_X.index.values.reshape(-1, 1)

        from sklearn.cluster import KMeans
        x = df_Y[z_serie].values.reshape(-1, 1)
        res = KMeans(n_clusters=n, random_state=0).fit(x)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')

        clr = np.array(['#2200CC', '#D9007E', '#FF6600', '#FFCC00', '#ACE600', '#0099CC',
                        '#8900CC', '#FF0000', '#FF9900', '#FFFF00', '#00CC01', '#0055CC'])

        y = df_X.values.sum(axis=1)
        ax.scatter(x, y, color=clr[res.labels_.astype(int)])
        ax.set_xlabel('Objective')
        ax.set_ylabel('Installed power')
        ax.set_title('Results clustering')

        fig.savefig(file_name + '_cluster.png')
        fig.savefig(file_name + '_cluster.eps')
    else:
        print('Not enough data for cluster plot')


def plot_3D(file_name, df_X, df_Y, x_serie='Solar (kW)', y_serie='Wind (kW)', z_serie='lcoe'):
    """

    :param file_name:
    :param df_X:
    :param df_Y:
    :param x_serie:
    :param y_serie:
    :param z_serie:
    :return:
    """
    # Make data.
    X = df_X[x_serie].values
    Y = df_X[y_serie].values
    Z = df_Y[z_serie].values

    # Scatter plot
    fig = plt.figure(figsize=(16, 8))
    ax = fig.gca(projection='3d')
    ax.scatter(X, Y, Z)
    ax.set_xlabel(x_serie)
    ax.set_ylabel(y_serie)
    ax.set_zlabel(z_serie)
    ax.set_facecolor('white')

    ending = '_' + x_serie + '_' + y_serie + '_' + z_serie
    fig.savefig(file_name + ending + '_scatter.png')
    fig.savefig(file_name + ending + '_scatter.eps')

    # Surface plot
    fig = plt.figure(figsize=(16, 8))
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True)
    ax.set_xlabel(x_serie)
    ax.set_ylabel(y_serie)
    ax.set_zlabel(z_serie)

    ending = x_serie + '_' + y_serie + '_' + z_serie
    fig.savefig(file_name + ending + '_surf.png')
    fig.savefig(file_name + ending + '_surf.eps')


if __name__ == '__main__':

    # load data
    fname = '/home/santi/Documentos/Bitbucket/spv_phd/Desarrollos/DesalinationPlant/tests_for_the_report/num 20 scatter/Space_mapping_num_20.xlsx'
    df_X = pd.read_excel(fname, sheet_name='X')
    df_Y = pd.read_excel(fname, sheet_name='Y')

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    # X = df_X['Solar (kW)'].values
    # Y = df_X['Wind (kW)'].values
    # # X, Y = np.meshgrid(X, Y)
    # Z = df_Y['f_lcoe'].values

    experiment_name = 'Space_mapping_num_20'

    # for obj in ['f_lcoe', 'f_cost', 'f_income',	'f_income_cost', 'f_benefit']:
    for obj in ['LCOE', 'Cost', 'Income', 'Income_Cost_Ratio', 'Benefit']:
        plot_3D(experiment_name, df_X, df_Y, x_serie='Solar (kW)', y_serie='Wind (kW)', z_serie=obj)
        plot_3D(experiment_name, df_X, df_Y, x_serie='Wind (kW)', y_serie='Battery (kWh)', z_serie=obj)
        plot_3D(experiment_name, df_X, df_Y, x_serie='Solar (kW)', y_serie='Battery (kWh)', z_serie=obj)
    plot_2D(experiment_name, df_X, df_Y)