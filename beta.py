import numpy as np
import warnings


def fgammaL(x):
    return 2. * x / (1. - 4. * x + np.sqrt((1. - 6. * x) * (1. - 2. * x)))

def W(beta,v):
    return np.diag([np.exp(-beta * eps_i) for eps_i in v])

def newton_raphson(fun, x, h=1e-8):
    """ Newton-Raphson's method for the argument iteration """
    warnings.filterwarnings('ignore') # to suppress warnings
    try:
        return  x - 2.0 * h * fun(x) / (fun(x+h) - fun(x-h))
    except ZeroDivisionError:
        return  x - 2.0 * h 

def zero_of_function(fun, x0=0.5, precision=10):
    """ finding the zero of function f(x) closest to x0 """ 
    h = 10.0**(-precision)
    x = 1.0
    x_new = x0
    while abs(x_new - x) > h:
        x = x_new
        x_new = newton_raphson(fun, x)
    return round(x_new, precision)  

class BetaCalc:
    """ Calculation of inverse temperature of adsorption"""
    def __init__(self, n, x = 1./6.):
        if n <= 0:
            raise ValueError("n must be > 0")
        self.n = n
        self.x = x
        self.P  = np.eye(self.n, self.n, k=1, dtype='float') 
        self.P[n-1,0] = 1.
        self.E = np.eye(n, n)
        v_L = np.zeros(n)
        v_L[1] = 1.
        v_L = np.fft.fft(v_L)
        Gv_L = fgammaL(v_L * self.x)
        u = np.fft.ifft(Gv_L).real
        self.G_L = np.array([np.roll(u, i) for i in range(n)])
        
    def beta(self, v, *kwargs):
        if len(v) != self.n:
            raise ValueError("input length v != input n in BetaCalc instance")
        R = lambda beta: self.P @ W(beta,v)
        f = lambda beta: np.linalg.det(self.E - 4.0 * self.x * R(beta) - self.x * self.G_L @ R(beta))
        return zero_of_function(f,*kwargs)

def auto_cor(v: list) -> np.ndarray:
    """ cyclic autocorrelation of the vector v """
    if v:
        return np.fft.ifft(np.fft.fft(v[::-1])*np.fft.fft(v))[::-1].real / len(v)
    else:
        return np.array([])


def random_chain_generation(n, values, limit=1000, mult_attempt=10):
    B = BetaCalc(n=n)
    attempt_limit = mult_attempt * limit
    err_counter = 0
    accept_counter = 0
    X, y = [], []
    possibilities = [1.0 / len(values)] * len(values)
    while True:
        v = np.random.choice(values, n, possibilities)
        b = B.beta(v)
        if np.isfinite(b) and b not in y:
            X.append(list(v))
            y.append(b)
            accept_counter += 1
            if accept_counter >= limit:
                return np.array(X), np.array(y)
        else:
            err_counter += 1
            if err_counter > attempt_limit:
                raise Exception("Attempt limit overflowed")


if __name__ == '__main__':

    B = BetaCalc(n=4, x=0.5)
    x = B.beta([-1.,0.,0.,0.])
    print(x)


