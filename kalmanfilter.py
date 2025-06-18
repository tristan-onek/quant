import numpy as np
import pandas as pd

class Kalman1D:
    """
    Minimal 1-dimensional Kalman filter for streaming prices
    State 'x' is latent “true” price & z is the observed (or noisy) price
    f = 1 (price follows random walk); h = 1 (observe price directly)
    """
    def __init__(self, process_var: float, meas_var: float,
                 x0: float | None = None, P0: float | None = None):
        self.Q = process_var            # process (model) variance
        self.R = meas_var               # measurement variance
        self.x = x0                     # state estimate
        self.P = P0 if P0 is not None else 1.0  #  covar

    def step(self, z: float) -> float:
        if self.x is None:              # bootstrap with first obs
            self.x = z
            return self.x

        # ── Predict
        x_pred = self.x                # F = 1 ⇒ x̂(k|k-1) = x(k-1)
        P_pred = self.P + self.Q       # P(k|k-1)

        # ── Update
        K = P_pred / (P_pred + self.R) # Kalman gain
        self.x = x_pred + K * (z - x_pred)
        self.P = (1 - K) * P_pred
        return self.x


def kalman_filter_series(prices: pd.Series,
                         process_var: float = 1e-3,
                         meas_var: float = 1e-1) -> pd.Series:
    kf = Kalman1D(process_var, meas_var)
    return prices.apply(kf.step)


# Example Use (synthetic walk and noise):
if __name__ == "__main__":
    n = 500
    true_price = 100 + np.cumsum(np.random.normal(0, 0.2, n))
    observed = true_price + np.random.normal(0, 1.0, n)  # noisy tape

    df = pd.DataFrame({
        "Observed": observed,
        "Kalman": kalman_filter_series(pd.Series(observed),
                                       process_var=0.04, meas_var=1.0)
    })
    print(df.head())
