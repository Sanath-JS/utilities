import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
#numpy==1.24.4
#matplotlib==3.7.4
#scipy==1.10.1

class LOD_CALC:
    def __init__(self, conc, intensity, sample_intensity=None, confidence_level=0.95):
        self.conc = np.array(conc, dtype=float)
        self.intensity = np.array(intensity, dtype=float)
        self.sample_intensity = np.array(sample_intensity, dtype=float) if sample_intensity is not None else np.array([])
        self.confidence_level = confidence_level

        self.coef = np.polyfit(self.conc, self.intensity, 1)
        self.slope = self.coef[0]
        self.intercept = self.coef[1]

        
        print(f"Slope = {self.slope:.6f}")
        print(f"Intercept = {self.intercept:.6f}")

        
        self.predicted_intensity = self.slope * self.conc + self.intercept
        self.n = len(self.conc)
        self.dof = self.n - 2
        self.mean_conc = np.mean(self.conc)

    def get_regression_std_error(self):
        residuals = self.intensity - self.predicted_intensity
        return np.sqrt(np.sum(residuals ** 2) / self.dof)

    def calculate_lod(self):
        sigma_y_x = self.get_regression_std_error()
        sum_sq_diff = np.sum((self.conc - self.mean_conc) ** 2)
        term = np.sqrt(1 + (self.mean_conc ** 2 / sum_sq_diff))
        lod_x = (3.3 * sigma_y_x * term) / self.slope
        lod_y = self.slope * lod_x + self.intercept

        print(f"LOD  = ({lod_x:.6f}, {lod_y:.6f})")
        return lod_x, lod_y

    def confidence_band(self):
        t_value = t.ppf((1 + self.confidence_level) / 2, self.dof)
        residual_std_error = self.get_regression_std_error()
        sum_sq_diff = np.sum((self.conc - self.mean_conc) ** 2)

        se_pred = residual_std_error * np.sqrt(
            1 / self.n + (self.conc - self.mean_conc) ** 2 / sum_sq_diff
        )
        margin = t_value * se_pred

        lower_band = self.predicted_intensity - margin
        upper_band = self.predicted_intensity + margin

        plt.fill_between(
            self.conc,
            lower_band,
            upper_band,
            color="blue",
            alpha=0.15,
            label=f"{int(self.confidence_level * 100)}% Confidence Band"
        )
        return lower_band, upper_band

    def three_sigma_band(self):
        std_dev = np.std(self.intensity, ddof=1)
        margin = 3 * std_dev
        lower = self.predicted_intensity - margin
        upper = self.predicted_intensity + margin

        plt.fill_between(
            self.conc,
            lower,
            upper,
            alpha=0.1,
            label="3σ Band"
        )
        return lower, upper

    def plot(self, show_lod=True, show_3sigma=False):
        poly_fn = np.poly1d(self.coef)

        # PLOT DATA AND FIT
        plt.plot(self.conc, self.intensity, 'bo', label='Experimental Data')
        plt.plot(self.conc, poly_fn(self.conc), '--b', label='Linear Fit')

        
        if self.sample_intensity.size > 0:
            sample_int = self.sample_intensity[0]
            sample_conc = (sample_int - self.intercept) / self.slope
            plt.scatter(sample_conc, sample_int, color='red', label='Sample')
            print(f"[Sample] Intensity = {sample_int:.6f}, Estimated Conc = {sample_conc:.6f}")

     
        self.confidence_band()

        
        if show_3sigma:
            self.three_sigma_band()

     
        if show_lod:
            lod_x, lod_y = self.calculate_lod()
            ylim = plt.ylim()
            plt.vlines(lod_x, ylim[0], lod_y, color='black', linestyles='dashed', label='LOD')
            plt.hlines(lod_y, 0, lod_x, color='black', linestyles='dashed')
            plt.text(lod_x, ylim[0] + 0.05 * (ylim[1] - ylim[0]), f'LOD = {lod_x:.3f}',
                     fontsize=8, color='black',
                     horizontalalignment='center')

      
        plt.xlabel(r'Concentration', fontsize=14)
        plt.ylabel("Intensity (a.u.)", fontsize=14)
        plt.xlim(0, max(self.conc) * 1.2)
        plt.ylim(0)
        plt.legend()
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    conc = [10, 20, 40, 100, 250]
    intensity = [70.719, 179.764, 194.010, 410.030, 1016.408]
    sample_intensity = None #[146.525]

    calc = LOD_CALC(conc, intensity, sample_intensity, confidence_level=0.95)
    calc.plot(show_lod=True, show_3sigma=False)

