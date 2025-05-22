from typing import Callable

import numpy as np
import numpy.typing as npt

from refmod.hapke.helper import h_function


class Hapke:
    def __init__(
        self,
        incidence_direction: npt.NDArray,
        emission_direction: npt.NDArray,
        surface_orientation: npt.NDArray,
        single_scattering_albedo: npt.NDArray,
        phase_function: Callable[[npt.NDArray], npt.NDArray],
        opposition_effect_h: float = 0,
        oppoistion_effect_b0: float = 0,
        roughness: float = 0,
    ):
        self.sun = incidence_direction
        self.cam = emission_direction
        self.normal = surface_orientation
        self.w = single_scattering_albedo
        self.p = phase_function
        self.h = opposition_effect_h
        self.b0 = oppoistion_effect_b0
        self.tb = roughness
