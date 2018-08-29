import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt


if __name__ == '__main__':

    # load data
    fname = '/home/santi/Documentos/Bitbucket/spv_phd/Desarrollos/DesalinationPlant/tests_for_the_report/Space_mapping/Space_mapping_num_25.xlsx'
    df_X = pd.read_excel(fname, sheet_name='X')
    df_Y = pd.read_excel(fname, sheet_name='Y')

    X = df_X.values
    Y = df_Y.values

    pca = PCA()
    new_X = pca.fit_transform(X, Y)

    print(new_X)
    print('Shape:', new_X.shape)

    # plt.plot(new_X)
    # plt.show()