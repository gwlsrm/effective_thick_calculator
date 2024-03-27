import os
import json
import math
import typing as tp
from dataclasses import dataclass, asdict
from functools import reduce

import numpy as np


EPSILON = 1e-10


class ElementMu:
    """
        mass attenuation coefficients from xcom
    """
    def __init__(self, z: int, symbol: str, energies: tp.List[float], mus: tp.List[float]) -> None:
        self.z = z
        self.symbol = symbol
        self.energies = np.log(np.array(energies))
        self.mus = np.log(np.array(mus))

    @staticmethod
    def read_from_file(filepath: str, col_num: int = 7) -> "ElementMu":
        _, filename = os.path.split(filepath)
        z, symbol = ElementMu.extract_element_from_filename(filename)
        energies = []
        mus = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                words = line.split()
                assert len(words) == 8
                values = [float(w) for w in words]
                energies.append(values[0])
                mus.append(values[col_num])

        return ElementMu(z, symbol, energies, mus)

    @property
    def name(self):
        return f'{self.z}{self.symbol}'

    @staticmethod
    def extract_element_from_filename(filename: str) -> tp.Tuple[int, str]:
        assert len(filename) in (4, 5)
        z = int(filename[:3])
        symbol = filename[3:]
        return z, symbol

    def get_mu(self, energy_mev: float) -> float:
        """
            returns mu for energy in MeV, units: cm^2/g
        """
        e = np.log(energy_mev)
        ri = np.searchsorted(self.energies, e, side='right')
        if ri == 0:
            ri += 1
        if ri == len(self.energies):
            ri -= 1
        li = ri - 1

        el = self.energies[li]
        er = self.energies[ri]
        mul = self.mus[li]
        mur = self.mus[ri]

        w = (e - el) / (er - el)

        mu = (1-w) * mul + w * mur
        return np.exp(mu)

    def __str__(self) -> str:
        return f"element {self.z}{self.symbol} with {len(self.mus)} mus"


class MuDB:
    def __init__(self, elements: tp.List[ElementMu]) -> None:
        self.elements = sorted(elements, key=lambda e: e.z)
        self.symbol_to_z = {e.symbol: e.z for e in elements}
        self.element_symbols = [e.symbol for e in elements]
        self.element_name_to_z = {e.name: e.z for e in elements}

    @staticmethod
    def read_from_directory(dir_name: str) -> "MuDB":
        files = sorted(
            [frec.path for frec in os.scandir(path=dir_name) if frec.is_file()])
        elements = [ElementMu.read_from_file(filepath) for filepath in files]
        return MuDB(elements)

    def __str__(self) -> str:
        return f"MuDB with {len(self.elements)} elements"

    def get_mu_by_z(self, z: int, energy_mev: float) -> float:
        return self.elements[z].get_mu(energy_mev)

    def get_mu_by_name(self, element_name: str, energy_mev: float) -> float:
        z = self.element_name_to_z[element_name]
        return self.get_mu_by_z(z, energy_mev)

    def get_mu_by_symbol(self, element_symbol: str, energy_mev: float) -> float:
        z = self.symbol_to_z[element_symbol]
        return self.get_mu_by_z(z, energy_mev)


@dataclass
class Element:
    z: int
    frac: float


class Material:
    def __init__(self, elements_mass_fraction: tp.List[Element]) -> None:
        self.elements_mass_fraction = elements_mass_fraction

    def _norm_mass_fractions(self) -> None:
        norm = reduce(lambda x,y: x + y.frac, self.elements_mass_fraction, 0)
        if math.isclose(norm, 0.0):
            raise RuntimeError("Wrong material")
        if not math.isclose(norm, 1.0):
            self.elements_mass_fraction = [Element(e.z, e.frac/norm) for e in self.elements_mass_fraction]

    @staticmethod
    def read_from_json(material_string: str) -> "Material":
        """
        format: {name: ..., rho: ..., elements: [{z:..., frac: ...}, ...]}
        """
        mat = json.loads(material_string)
        el = mat["elements"]
        elements_mass_fraction = [Element(e['z'], e['frac']) for e in el]
        m = Material(elements_mass_fraction)
        m._norm_mass_fractions()
        return m

    @staticmethod
    def read_from_sl_json(material_string: str) -> "Material":
        """
        format: {Name: ..., Ro: ..., Compound: [{"z": frac}, ...]}
        """
        mat = json.loads(material_string)
        el = mat["Compound"]
        elements_mass_fraction = []
        for e in el:
            element = [Element(int(z), frac) for z, frac in e.items()][0]
            elements_mass_fraction.append(element)
        m = Material(elements_mass_fraction)
        m._norm_mass_fractions()
        return m

    def __str__(self) -> str:
        return json.dumps([asdict(e) for e in self.elements_mass_fraction])


def get_material_mu(material: Material, mu_db: MuDB, energy_mev: float) -> float:
    mu = 0.0
    norm = 0.0
    for e in material.elements_mass_fraction:
        mu += e.frac * mu_db.get_mu_by_z(e.z, energy_mev)
        norm += e.frac
    return mu / (norm + EPSILON)


def get_material_mus(material: Material, mu_db: MuDB, energies_mev: tp.List[float]) -> tp.List[float]:
    return [get_material_mu(material, mu_db, e) for e in energies_mev]


def main():
    mu_db = MuDB.read_from_directory("XCOM/")
    print(mu_db)
    # for e in mu_db.elements:
    #     print(e)
    print(mu_db.get_mu_by_name("82Pb", 0.92))
    m = Material.read_from_json("""
        {
            "Name": "Non-corrosive steel",
            "rho": 7.93,
            "elements":[
                {"z": 24, "frac": 0.001},
                {"z": 26, "frac": 0.971},
                {"z": 28, "frac": 0.018},
                {"z": 22, "frac": 0.01}
            ]
        }
    """)
    print(m)
    print(get_material_mu(m, mu_db, 0.92))


if __name__ == "__main__":
    main()