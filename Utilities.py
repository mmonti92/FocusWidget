import numpy as np
import os
import lmfit as fit
import PIL as pil
import warnings as wn
import dataclasses as dc
import DataAnalysis.Models as mod
import DataAnalysis.ReadWriteFunctions as rw
import DataAnalysis.FittingFunctions as ff


@dc.dataclass(slots=True)
class Camera:
    name: str
    px_size: float
    center: tuple


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
        "Attention! The jpg and tif images are not accurate!",
        category=UserWarning,
    )
    if len(np.shape(pil.Image.open(file))) == 3:
        img = np.array(pil.Image.open(file).convert("L")).astype(np.float32)
    elif len(np.shape(pil.Image.open(file))) == 2:
        img = np.array(pil.Image.open(file)).astype(np.float32)
    else:
        raise RuntimeError("Wrong data shape")

    return np.transpose(img)


def GetReader(file: str):
    READERS = {
        ".csv": Readcsv,
        ".dat": Readdat,
        ".jpg": Readjpgtif,
        ".tif": Readjpgtif,
        ".tiff": Readjpgtif,
    }
    _, ext = os.path.splitext(file)
    return READERS[ext.lower()]


def GetCamera(size: int):
    if size > 2000:
        return Camera("FemtoEasy", 5.48, (7000.0, 7000.0))
    elif 0 < size < 2000:
        return Camera("THzRigi", 25.0, (1500.0, 2000.0))
    else:
        raise ValueError("Wrong size")


def ReadImg(file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reader = GetReader(file)
    img = reader(file)
    cam = GetCamera(img.shape[0])

    x = (
        np.linspace(0, np.shape(img)[1] * cam.px_size, np.shape(img)[1])
        - cam.center[0]
    )
    y = (
        np.linspace(0, np.shape(img)[0] * cam.px_size, np.shape(img)[0])
        - cam.center[1]
    )

    return x, y, img


def FitSpot(
    img: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[fit.minimizer.MinimizerResult, np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(x, y)

    x_min, x_max = min(x[0], x[-1]), max(x[0], x[-1])
    y_min, y_max = min(y[0], y[-1]), max(y[0], y[-1])

    par = fit.Parameters()
    par.add("A", np.amax(img) / 2, min=0)
    par.add("t", 0, vary=True)  # min=0, max=np.pi / 2)
    par.add("sx", 200, min=0)
    par.add("sy", 200, min=0)
    par.add("x0", 0, min=x_min, max=x_max)
    par.add("y0", 0, min=y_min, max=y_max)
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
