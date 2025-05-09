from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from dataclasses import dataclass
from stim import DetectorErrorModel, DemTarget, DemTargetWithCoords
from scipy.sparse import csc_matrix
import numpy as np


class Basis(Enum):
    Z = 0
    X = 1


@dataclass(frozen=True)
class CircuitLevelError:
    dets: tuple[DemTargetWithCoords]
    obs: tuple[DemTarget]
    basis: Basis


@dataclass
class CheckMatrices:
    check_map: csc_matrix
    check_matrix: csc_matrix
    obs_matrix: csc_matrix
    priors: np.ndarray


def _det_basis_coord(det: DemTargetWithCoords) -> Basis:
    """
    Returns the basis of the detector based on the 4th coordinate (0 == Z, 1 == X)
    """
    return Basis(det.coords[3])


def _det_basis_parity(det: DemTargetWithCoords) -> Basis:
    return Basis.Z if (det.coords[0] % 4 + det.coords[1] % 4) % 4 == 0 else Basis.X


def _prior_dict_to_matrices(
    prior_dict: dict[CircuitLevelError, float], num_detectors: int
) -> CheckMatrices:
    det_list: list[DemTarget] = []
    det_map: dict[DemTarget, int] = {}
    det_row_idx: list[int] = []
    det_col_idx: list[int] = []

    obs_list: list[DemTarget] = []
    obs_map: dict[DemTarget, int] = {}
    obs_row_idx: list[int] = []
    obs_col_idx: list[int] = []

    priors: list[float] = []

    for i, (c_err, prior) in enumerate(prior_dict.items()):
        priors.append(prior)

        for det in c_err.dets:
            det_val = det.dem_target.val
            if not det in det_list:
                det_map[det_val] = len(det_list)
                det_list += [det]
            det_row_idx += [det_map[det_val]]
            det_col_idx += [i]

        for obs in c_err.obs:
            if not obs in obs_list:
                obs_map[obs] = len(obs_list)
                obs_list += [obs]
            obs_row_idx += [obs_map[obs]]
            obs_col_idx += [i]

    # Resulting check matrix may have fewer dets than original
    check_map = csc_matrix(
        (np.ones(len(det_map)), (list(det_map.values()), list(det_map.keys()))),
        shape=(len(det_list), num_detectors),
    )

    check_matrix = csc_matrix(
        (np.ones(len(det_row_idx)), (det_row_idx, det_col_idx)),
        shape=(len(det_list), len(prior_dict)),
    )
    obs_matrix = csc_matrix(
        (np.ones(len(obs_row_idx)), (obs_row_idx, obs_col_idx)),
        shape=(len(obs_list), len(prior_dict)),
    )

    return CheckMatrices(check_map, check_matrix, obs_matrix, np.array(priors))


def detector_error_model_to_css_checks(
    dem: DetectorErrorModel,
    fn_det_basis: Callable[[DemTargetWithCoords], Basis] = _det_basis_coord,
) -> tuple[CheckMatrices, CheckMatrices]:
    """
    Convert a DetectorErrorModel into separate Z/X check matrices

    args:
        dem: DetectorErrorModel
            The detector error model to convert
        fn_det_basis: Callable[[DemTargetWithCoords], Basis]
            A function that takes a detector and returns the basis of the CSS stabilizer it checks (Z/X)
            By default, the 4th coordinate of the detector is used to determine the basis (0 == Z, 1 == X)
    returns:
        tuple[CheckMatrices, CheckMatrices]
            The Z and X check matrices
    """
    det_coords: dict[int, list[float]] = dem.get_detector_coordinates()

    x_error_priors: dict[CircuitLevelError, float] = {}
    z_error_priors: dict[CircuitLevelError, float] = {}
    for instr in dem.flattened():
        if instr.type == "error":
            prior = instr.args_copy()[0]
            x_dets: list[DemTarget] = []
            z_dets: list[DemTarget] = []
            obs: list[DemTarget] = []
            for targ in instr.targets_copy():
                if targ.is_relative_detector_id():
                    det = DemTargetWithCoords(
                        dem_target=targ, coords=det_coords[targ.val]
                    )
                    basis = fn_det_basis(det)
                    if basis == Basis.Z:
                        z_dets.append(det)
                    elif basis == Basis.X:
                        x_dets.append(det)
                    else:
                        raise ValueError(f"Invalid basis: {basis}")
                elif targ.is_logical_observable_id():
                    obs.append(targ)

            if len(z_dets) > 0:
                z_error = CircuitLevelError(tuple(z_dets), tuple(obs), Basis.Z)
                z_error_priors[z_error] = z_error_priors.setdefault(z_error, 0) + prior

            if len(x_dets) > 0:
                x_error = CircuitLevelError(tuple(x_dets), tuple(obs), Basis.X)
                x_error_priors[x_error] = x_error_priors.setdefault(x_error, 0) + prior

    z_check_matrices = _prior_dict_to_matrices(z_error_priors, dem.num_detectors)
    x_check_matrices = _prior_dict_to_matrices(x_error_priors, dem.num_detectors)

    return z_check_matrices, x_check_matrices
