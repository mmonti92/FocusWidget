import numpy as np
import lmfit as fit

import DataAnalysis.Models as mod
import DataAnalysis.ReadWriteFunctions as rw
import DataAnalysis.FittingFunctions as ff


def ReadImg(file):
    img = rw.Reader(file, delimiter=",")
    # print(np.shape(img))
    x = np.linspace(0, np.shape(img)[1] * 25, np.shape(img)[1]) - 1500
    y = np.linspace(0, np.shape(img)[0] * 25, np.shape(img)[0]) - 2000

    return x, y, img


def FitSpot(img, x, y):
    X, Y = np.meshgrid(x, y)
    par = fit.Parameters()
    par.add("A", np.amax(img) / 2, min=0)
    par.add("t", -1, vary=True)  # min=0, max=np.pi / 2)
    par.add("sx", 200, min=0)
    par.add("sy", 200, min=0)
    par.add("x0", 360, min=-2500, max=2500)
    par.add("y0", 360, min=-2500, max=2500)
    par.add("C", img[0, 0])
    guess = mod.GeneralGauss2D(par, X, Y)
    res = ff.Residual_wrapper("GeneralGauss2D")
    out = fit.minimize(
        res,
        par,
        args=(X, Y),
        kws={"data": img},
        nan_policy="omit",
    )
    fitted = mod.GeneralGauss2D(out.params, X, Y)

    return out, guess, fitted


if __name__ == "__main__":
    ...
