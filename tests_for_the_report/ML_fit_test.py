import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor

if __name__ == '__main__':

    fname_train = './Optimization_mapping/DYCORS/Space_mapping_DYCORS_5000.xlsx'
    fname_test = './Space_mapping/Space_mapping_num_25.xlsx'

    df_X = pd.read_excel(fname_train, sheet_name='X')

    X_train = pd.read_excel(fname_test, sheet_name='X').values
    Y_train = pd.read_excel(fname_test, sheet_name='Y')['Benefit'].values

    X_test = pd.read_excel(fname_train, sheet_name='X').values
    Y_test = pd.read_excel(fname_train, sheet_name='Y')['Benefit'].values

    # model = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=-1)

    model = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                metric='minkowski', metric_params=None, n_jobs=-1)

    scaler = RobustScaler()

    model.fit(scaler.fit_transform(X_train), Y_train)

    y_predict = model.predict(scaler.transform(X_test))

    mse = np.sqrt(np.power(y_predict - Y_test, 2).sum())
    mape = np.mean(np.abs((Y_test - y_predict) / Y_test)) * 100

    print(mse, '€, ', mape, '% error')

    # plot sensitivity
    fig1 = plt.figure(figsize=(12, 6))
    # fig2 = plt.figure(figsize=(12, 6))

    for i in range(3):
        x = X_test[:, i]
        benef = Y_test

        ax1 = fig1.add_subplot(2, 2, i + 1)
        # ax2 = fig2.add_subplot(2, 2, i + 1)

        name = df_X.columns[i]

        df1 = pd.DataFrame(data=-Y_test, index=x, columns=[name + ' real']).sort_index()
        df2 = pd.DataFrame(data=-y_predict, index=x, columns=[name + ' predicted']).sort_index()
        ax1.set_ylabel('Benefit')
        # ax2.set_ylabel('Benefit')

        df1.plot(marker='.', linestyle='None', ax=ax1, alpha=0.6)
        df2.plot(marker='.', linestyle='None', ax=ax1, alpha=0.6)

    ax1 = fig1.add_subplot(2, 2, 4)
    ax1.plot(Y_test, label='Real')
    ax1.plot(y_predict, label='Predicted')
    ax1.set_title('Fit')
    ax1.legend()


    plt.show()


