import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score
from statsmodels.tsa.api import VAR, AutoReg

if __name__ == "__main__":

    data_source = "torch_celled"
    station_name = "gotnya"

    if data_source == "meteo_ru":
        # data from meteo-ru
        meteoru = pd.read_csv(
            f"data/meteo_ru/{station_name}.txt", delimiter=";", header=None
        )
        meteoru.columns = ["id", "year", "month", "day", "temper", "precip"]
        meteoru.drop(["id"], axis=1, inplace=True)

        meteoru["temper"] = pd.to_numeric(meteoru["temper"], errors="coerce")
        meteoru["precip"] = pd.to_numeric(meteoru["precip"], errors="coerce")
        meteoru["precip"] = np.where(meteoru["temper"] > 10, meteoru["precip"], 0)
        meteoru["temper"] = np.where(meteoru["temper"] > 10, meteoru["temper"], 0)
        meteoru.dropna(how="any", inplace=True)
        meteoru.drop(["month", "day"], axis=1, inplace=True)
        gtk = meteoru.groupby(["year"]).sum()

        # filling NaN value if there any
        gtk["temper"] = gtk["temper"].replace(to_replace=0.0, method="ffill")
        gtk["precip"] = gtk["precip"].replace(to_replace=0.0, method="ffill")

        gtk["value"] = 10 * gtk["precip"] / gtk["temper"]
        gtk.to_csv("data/meteo_ru/gtk_belg.csv")

        plt.plot(gtk.index, gtk.value)
        plt.title(
            f"ГТК метеостанции {station_name} (Белгородская область), \n усреднение по месяцам"
        )
        # plt.xlabel('Year')
        # plt.ylabel('GTK')
        plt.savefig("results/gtk_belg.png", dpi=300)
        gtk_aver = np.array(gtk.value.to_list())
        temp_aver = np.array(gtk.temper.to_list())
        precip_aver = np.array(gtk.precip.to_list())

    elif data_source == "torch_celled":
        # parameters for torch celled data
        N_CELLS_HOR = 9
        N_CELLS_VER = 16

        precip_data = np.array(
            torch.load(
                "data/celled_data/precip_celled_data_"
                + str(N_CELLS_HOR)
                + "x"
                + str(N_CELLS_VER)
            )
        )
        precip_data = np.squeeze(precip_data, axis=1)
        temper_data = np.array(
            torch.load(
                "data/celled_data/tmean_celled_data_"
                + str(N_CELLS_HOR)
                + "x"
                + str(N_CELLS_VER)
            )
        )
        temper_data = np.squeeze(temper_data, axis=1)

        # data filtering
        precip_data_filtr = np.where(temper_data > 10, precip_data, 0)
        temper_data_filtr = np.where(temper_data > 10, temper_data, 0)
        # yearly rolling sum
        precip_annual_rolling = rolling_mean_along_axis(precip_data_filtr, W=12, axis=0)
        temper_annual_rolling = rolling_mean_along_axis(temper_data_filtr, W=12, axis=0)
        # yearly sum
        # leave up to 12*x months
        precip_data_filtr = precip_data_filtr[:756, :, :]
        temper_data_filtr = temper_data_filtr[:756, :, :]

        temper_data_filtr = temper_data_filtr.reshape(63, -1, N_CELLS_HOR, N_CELLS_VER)
        precip_data_filtr = precip_data_filtr.reshape(63, -1, N_CELLS_HOR, N_CELLS_VER)

        precip_annual = np.sum(precip_data_filtr, axis=1)
        temper_annual = np.sum(temper_data_filtr, axis=1)

        gtk = 10 * precip_annual / temper_annual

        # average over x and y
        gtk_aver = gtk.mean(axis=(1, 2))

    # Split train-test

    total_len = gtk_aver.shape[0]
    train_test_split = 0.8
    train_len = int(train_test_split * total_len)
    test_len = total_len - train_len
    gtk_train = gtk_aver[:train_len]
    gtk_test = gtk_aver[train_len:]

    # Constant forecast

    gtk_constant = [gtk_train[-1] for i in range(len(gtk_test))]
    print("Constant baseline: ")
    print("R2: ", r2_score(gtk_test, gtk_constant))
    print("MAPE: ", mape(gtk_test, gtk_constant))

    plt.plot(gtk.index, gtk.value)
    plt.plot(gtk.index[train_len:], gtk_constant, label="Constant forecast")
    plt.axvline(x=gtk.index[train_len], color="k", linestyle="--")
    plt.title(
        f"ГТК метеостанции {station_name} (Белгородская область), \n усреднение по месяцам"
    )
    plt.legend()
    plt.savefig("results/gtk_constant.png", dpi=300)

    # Rolling average

    best_R2 = -3
    n_lags = -1

    for i in range(1, 25):

        gtk_rolling = running_mean_convolve(gtk_train, lags=i, test_len=test_len)
        if r2_score(gtk_test, gtk_rolling) > best_R2:
            n_lags = i
            best_R2 = r2_score(gtk_test, gtk_rolling)
            best_mape = mape(gtk_test, gtk_rolling)
            rolling_forecast = gtk_rolling

    print("n of lags: ", n_lags)
    print("R2: ", best_R2)
    print("MAPE: ", best_mape)

    plt.plot(gtk.index, gtk.value)
    plt.plot(gtk.index[train_len:], rolling_forecast, label="Rolling average forecast")
    plt.axvline(x=gtk.index[train_len], color="k", linestyle="--")
    plt.title(
        f"ГТК метеостанции {station_name} (Белгородская область), \n усреднение по месяцам"
    )
    plt.legend()
    plt.savefig("results/gtk_rolling.png", dpi=300)

    # AR(p) model

    best_R2 = -3
    n_lags = -1

    for i in range(1, 15):
        ar_model = AutoReg(gtk_train, lags=i, trend="n").fit()
        gtk_ar = ar_model.predict(start=train_len, end=total_len - 1, dynamic=False)
        if r2_score(gtk_test, gtk_ar) > best_R2:
            n_lags = i
            best_R2 = r2_score(gtk_test, gtk_rolling)
            best_mape = mape(gtk_test, gtk_rolling)
            ar_forecast = gtk_ar

    print("n of lags: ", n_lags)
    print("R2: ", best_R2)
    print("MAPE: ", best_mape)

    plt.plot(gtk.index, gtk.value)
    plt.plot(gtk.index[train_len:], ar_forecast, label="AR forecast")
    plt.axvline(x=gtk.index[train_len], color="k", linestyle="--")
    plt.title(
        f"ГТК метеостанции {station_name} (Белгородская область), \n усреднение по месяцам"
    )
    plt.legend()

    plt.savefig("results/gtk_ar.png", dpi=300)

    if data_source == "meteo_ru":
        # VAR model
        # Data for VAR
        weather_var = np.transpose(np.array([gtk_aver, temp_aver, precip_aver]))

        total_len = weather_var.shape[0]
        train_test_split = 0.8
        train_len = int(train_test_split * total_len)
        test_len = total_len - train_len
        weather_var_train = weather_var[:train_len]
        weather_var_test = weather_var[train_len:]

        best_R2 = -3
        n_lags = -1

        model_fitted = VAR(weather_var_train)
        for i in range(1, 15):
            results = model_fitted.fit(i)
            gtk_var = results.forecast(y=weather_var_train, steps=test_len)
            if r2_score(weather_var_test, gtk_var) > best_R2:
                n_lags = i
                best_R2 = r2_score(weather_var_test, gtk_var)
                best_mape = mape(weather_var_test, gtk_var)
                var_forecast = gtk_var

        print("n of lags: ", n_lags)
        print("R2: ", best_R2)
        print("MAPE: ", best_mape)

        plt.plot(gtk.index, gtk.value)
        plt.plot(gtk.index[train_len:], var_forecast[:, 0], label="VAR forecast")
        plt.axvline(x=gtk.index[train_len], color="k", linestyle="--")
        plt.title(
            f"ГТК метеостанции {station_name} (Белгородская область), \n усреднение по месяцам"
        )
        plt.legend()

        plt.savefig("results/gtk_var.png", dpi=300)
