import numpy as np

def rk4(model, motion, dt=None):
    t = motion.t
    dt = np.min(np.diff(t)) if dt is None else dt

    def f(tt, y):
        return model.rhs(y, motion.a(tt), motion.ad(tt), motion.V)

    y = model.y0(motion.a(t[0]), motion.V)
    cl, cd, Y = [], [], []
    tt = float(t[0])
    for t_out in t:
        while tt < t_out - 1e-14:
            h = min(dt, t_out - tt)
            k1 = f(tt, y)
            k2 = f(tt + h / 2, y + h / 2 * k1)
            k3 = f(tt + h / 2, y + h / 2 * k2)
            k4 = f(tt + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            tt += h
        a, ad = motion.a(t_out), motion.ad(t_out)
        c_l, c_d = model.coeffs(y, a, ad, motion.V)
        cl.append(c_l); cd.append(c_d); Y.append(y.copy())

    out = {"t": t, "alpha": motion.alpha, "C_L": np.array(cl), "C_D": np.array(cd)}
    Y = np.array(Y)
    for j, nm in enumerate(model.names):
        out[nm] = Y[:, j]
    return out
