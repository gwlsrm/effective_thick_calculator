"""
search optimal effective source thick for task: calculate efficiency to another material
"""
import argparse
import math
import typing as tp
from copy import copy
from dataclasses import dataclass

import numpy as np
import scipy.optimize as opt

from efaparser import Efficiency, EffPoint, get_efficiency_from_efr
from mu import Material, MuDB, get_material_mu, get_material_mus

EPSILON = 1e-10


@dataclass
class SelfAbsorptionCalculator:
    """SelfAbsorptionCalculator -- Wrapper for recalc_eff_point, it stores repeating init params"""
    source_material: Material
    source_rho: float
    target_material: Material
    target_rho: float
    thick: float
    dthick: float
    mu_db: MuDB

    def recalc_eff_point(self, energy: float, eff: float, deff: float) -> tp.Tuple[float, float]:
        """
        recalc efficiency point to another material using formula:
        eff(E) = eff_{0}(E) \frac{(1 - e^{-\mu(E) \rho t})\mu_0(E) \rho_0}{(1 - e^{-\mu_0(E) \rho_0 t}) \mu(E) \rho}
        """
        if math.isclose(self.thick, 0, abs_tol=1e-10):
            return eff, deff

        source_mu = get_material_mu(self.source_material, self.mu_db, energy / 1000.0)
        target_mu = get_material_mu(self.target_material, self.mu_db, energy / 1000.0)

        new_eff = eff * (
            ((1.0 - math.exp(-target_mu * self.target_rho * self.thick)) * (source_mu * self.source_rho)) /
            ((1.0 - math.exp(-source_mu * self.source_rho * self.thick)) * (target_mu * self.target_rho) + EPSILON)
        )
        # TODO: recalc deff from dthick
        return new_eff, deff


def recalc_efficiency_points_to_material(
        efficiency_points: tp.List[EffPoint], source_material: Material, source_rho: float,
        target_material: Material, target_rho: float, mu_db: MuDB,
        thick: float, dthick: float) -> tp.List[EffPoint]:
    """
        recalc efficiency points to another material (from source to target)
    """
    # create calculator
    calculator = SelfAbsorptionCalculator(
        source_material, source_rho, target_material, target_rho, thick, dthick, mu_db)
    # recalc
    new_efficiency_points = []
    for p in efficiency_points:
        new_p = copy(p)
        new_eff, new_deff = calculator.recalc_eff_point(p.energy, p.eff, p.deff)
        new_p.eff = new_eff
        new_p.deff = new_deff
        new_efficiency_points.append(new_p)
    return new_efficiency_points


def recalc_efficiency_to_material(
        efficiency: Efficiency, target_material: Material, target_rho: float, mu_db: MuDB,
        thick: tp.Optional[float] = None, dthick: float = 0.0) -> Efficiency:
    # prepare input params
    source_material = Material.read_from_sl_json(efficiency.header["Material"])
    source_rho = float(efficiency.header["Density,g/cm3"])
    if thick is None:
        thick = float(efficiency.header["Thick,mm"])/10.0
        dthick = float(efficiency.header["DThick,mm"])/10.0

    new_efficiency_points = recalc_efficiency_points_to_material(
        efficiency.points, source_material, source_rho, target_material, target_rho, mu_db,
        thick, dthick)

    return Efficiency(efficiency.name, efficiency.header_lines, new_efficiency_points)


@dataclass
class SelfAbsorptionCalculatorWithSetMu:
    """
    SelfAbsorptionCalculator -- Wrapper for recalc_eff_point, it stores repeating init params.
    Energy range must be fixed for this calculator.
    """
    source_material: Material
    source_rho: float
    target_material: Material
    target_rho: float
    thick: float
    dthick: float
    source_mus: tp.List[float]
    target_mus: tp.List[float]

    def recalc_eff_point(self, idx: int, eff: float, deff: float) -> tp.Tuple[float, float]:
        """
        recalc efficiency point to another material using formula:
        eff(E) = eff_{0}(E) \frac{(1 - e^{-\mu(E) \rho t})\mu_0(E) \rho_0}{(1 - e^{-\mu_0(E) \rho_0 t}) \mu(E) \rho}
        energy range is fixed in ctor, idx -- index in energy range
        """
        if math.isclose(self.thick, 0.0, abs_tol=1e-10):
            return eff, deff

        source_mu = self.source_mus[idx]
        target_mu = self.target_mus[idx]

        new_eff = eff * (
            ((1.0 - math.exp(-target_mu * self.target_rho * self.thick)) * (source_mu * self.source_rho)) /
            ((1.0 - math.exp(-source_mu * self.source_rho * self.thick)) * (target_mu * self.target_rho) + EPSILON)
        )
        # TODO: recalc deff from dthick
        return new_eff, deff


def recalc_efficiency_points_to_material_with_mus(
        efficiency_points: tp.List[EffPoint], source_material: Material, source_rho: float,
        target_material: Material, target_rho: float,
        source_mus: tp.List[float], target_mus: tp.List[float],
        thick: float, dthick: float) -> tp.List[EffPoint]:
    """
        recalc efficiency points to another material (from source to target)
    """
    # create calculator
    calculator = SelfAbsorptionCalculatorWithSetMu(
        source_material, source_rho, target_material, target_rho, thick, dthick,
        source_mus, target_mus)
    # recalc
    new_efficiency_points = []
    for i, p in enumerate(efficiency_points):
        new_p = copy(p)
        new_eff, new_deff = calculator.recalc_eff_point(i, p.eff, p.deff)
        new_p.eff = new_eff
        new_p.deff = new_deff
        new_efficiency_points.append(new_p)
    return new_efficiency_points


def _eff_points_to_numpy(eff_points: tp.List[EffPoint]) -> np.ndarray:
    return np.array([p.eff for p in eff_points])


class Loss:
    """
    L2 loss to compare 2 efficiencies, one efficiency is parameterized with thick
    """
    def __init__(self, source_efficiency: Efficiency, target_efficiency: Efficiency, mu_db: MuDB,
                 alpha: float = 0.0) -> None:
        # source
        self.source_efficiency = source_efficiency
        self.source_material = Material.read_from_sl_json(source_efficiency.header["Material"])
        self.source_rho = float(source_efficiency.header["Density,g/cm3"])
        self.source_mus = get_material_mus(
            self.source_material, mu_db, [p.energy/1000.0 for p in source_efficiency.points])
        # target
        self.target_efficiency = target_efficiency
        self.target_material = Material.read_from_sl_json(target_efficiency.header["Material"])
        self.target_rho = float(target_efficiency.header["Density,g/cm3"])
        self.target_mus = get_material_mus(
            self.target_material, mu_db, [p.energy/1000.0 for p in target_efficiency.points])
        self.target_effpoints = _eff_points_to_numpy(self.target_efficiency.points)

        self.mu_db = mu_db
        self.alpha = alpha

    def calc_loss(self, thick: float):
        recalc_efficiency_points = recalc_efficiency_points_to_material_with_mus(
            self.source_efficiency.points, self.source_material, self.source_rho,
            self.target_material, self.target_rho, self.source_mus, self.target_mus, thick, 0.0)
        recalc_effpoints = _eff_points_to_numpy(recalc_efficiency_points)
        loss = np.mean((recalc_effpoints - self.target_effpoints)**2) + self.alpha * thick
        return loss

    def calc_relative_dev(self, thick: float):
        recalc_efficiency_points = recalc_efficiency_points_to_material_with_mus(
            self.source_efficiency.points, self.source_material, self.source_rho,
            self.target_material, self.target_rho, self.source_mus, self.target_mus, thick, 0.0)
        recalc_effpoints = _eff_points_to_numpy(recalc_efficiency_points)
        ave_eff_points = (recalc_effpoints + self.target_effpoints) / 2.0
        dev = np.mean(
            ((recalc_effpoints - self.target_effpoints) / (ave_eff_points + EPSILON))**2
        )
        return np.sqrt(dev)


def main():
    parser = argparse.ArgumentParser(
        description="effective_thick_calculator: it finds optimal source thick for efficiency recalculation to another material")
    parser.add_argument("source_efficiency_filename", help="filename with source efficiency with material 1")
    parser.add_argument("target_efficiency_filename", help="filename with target efficiency with material 2")
    parser.add_argument("--path_to_xcom", "-x", default="XCOM", help="path to XCOM files with mass attenuation factors")
    parser.add_argument("--start_thick", "-t", type=float, default=0.0, help="start thickness in cm to search, it is usually close to real thickness")
    parser.add_argument("--alpha", type=float, default=0.0, help="regularization parameter to limit calculated thick")
    # parser.add_argument("--save_converted_efficiency to efr", action="store_true")
    args = parser.parse_args()

    # input effs
    eff_1 = get_efficiency_from_efr(args.source_efficiency_filename)
    eff_2 = get_efficiency_from_efr(args.target_efficiency_filename)
    mu_db = MuDB.read_from_directory(args.path_to_xcom)

    # loss calculator
    loss_calculator = Loss(eff_1, eff_2, mu_db)

    # scipy
    opt_t = opt.minimize(
        lambda x: loss_calculator.calc_loss(x[0]), x0=[args.start_thick], method='powell',
        # options={'xatol': 1e-6, 'disp': True}
    )
    opt_t = opt_t.x[0]
    deviation = loss_calculator.calc_relative_dev(opt_t)
    print(f"optimal t: {opt_t:.2f}: mean relative efficiency deviation: {deviation:.3f}")


if __name__ == "__main__":
    main()