import pandas as pd
from matplotlib import pyplot as plt


if __name__ == '__main__':

    # fname = '.\\Optimization_mapping\\DYCORS\\Space_mapping_DYCORS_5000.xlsx'
    # fname = '.\\Space_mapping\\Space_mapping_num_25.xlsx'
    # fname = '.\\Optimization_mapping\\GA\\Space_mapping_GA_5000.xlsx'
    fname = '.\\Optimization_mapping\\GradientBoostTree\\Space_mapping_GradientBoostTree_5000.xlsx'

    df_X = pd.read_excel(fname, sheet_name='X')
    df_Y = pd.read_excel(fname, sheet_name='Y')

    fig1 = plt.figure(figsize=(12, 6))
    fig2 = plt.figure(figsize=(12, 6))

    for i in range(3):

        x = df_X.values[:, i]
        lcoe = df_Y.values[:, 0]  # LCOE
        benef = df_Y.values[:, 4]  # Benefit

        ax1 = fig1.add_subplot(2, 2, i + 1)
        ax2 = fig2.add_subplot(2, 2, i + 1)

        name = df_X.columns[i]

        df1 = pd.DataFrame(data=-benef, index=x, columns=[name]).sort_index()
        df2 = pd.DataFrame(data=lcoe, index=x, columns=[name]).sort_index()

        ax1.set_ylabel('Benefit')
        ax2.set_ylabel('LCOE')

        df1.plot(marker='.', linestyle='None', ax=ax1, alpha=0.6)
        df2.plot(marker='.', linestyle='None', ax=ax2, alpha=0.6)

    plt.show()