import numpy as np
import os
import lmfit as fit
import matplotlib.pyplot as plt
import PIL as pil
import warnings as wn
import DataAnalysis.Models as mod
import DataAnalysis.ReadWriteFunctions as rw
import DataAnalysis.FittingFunctions as ff


def Readcsv(file: str) -> np.ndarray:
    return rw.Reader(file, delimiter=",")
    # return img


def Readdat(file: str) -> np.ndarray:
    img = (
        np.fromfile(file, dtype=np.uint16)
        .reshape(120, 160)
        .astype(np.float32)
    )
    return np.transpose(img)


def Readjpgtif(file: str) -> np.ndarray:
    wn.warn(
        "Attention! The jpg and tif images are lacking not accurate!",
        category=UserWarning,
    )
    img = np.array(pil.Image.open(file).convert("L")).astype(np.float32)
    return np.transpose(img)


def GetReader(file: str):
    READERS = {
        ".csv": Readcsv,
        ".dat": Readdat,
        ".jpg": Readjpgtif,
        ".tif": Readjpgtif,
    }
    _, ext = os.path.splitext(file)
    return READERS[ext.lower()]


def ReadImg(file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reader = GetReader(file)
    img = reader(file)

    x = np.linspace(0, np.shape(img)[1] * 25, np.shape(img)[1]) - 1500
    y = np.linspace(0, np.shape(img)[0] * 25, np.shape(img)[0]) - 2000

    return x, y, img


def FitSpot(
    img: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[fit.minimizer.MinimizerResult, np.ndarray, np.ndarray]:
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
