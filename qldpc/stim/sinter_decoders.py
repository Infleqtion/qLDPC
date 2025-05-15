from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any, Union

import numpy as np
import numpy.typing as npt
from ldpc.bp_decoder import BpDecoder
from ldpc.bplsd_decoder import BpLsdDecoder
from ldpc.bposd_decoder import BpOsdDecoder
from sinter import CompiledDecoder, Decoder
from stim import DemTargetWithCoords, DetectorErrorModel

from qldpc.objects import Pauli, PauliXZ
from qldpc.stim.util import (
    CheckMatrices,
    _det_basis_coord,
    detector_error_model_to_css_checks,
)

type BpDecoderType = Union[BpDecoder, BpLsdDecoder, BpOsdDecoder]
type BpDecoderClassType = Union[type[BpDecoder], type[BpLsdDecoder], type[BpOsdDecoder]]


class CompiledBPTypeDecoder(CompiledDecoder):
    def __init__(self, check_matrices: CheckMatrices, decoder: BpDecoderType) -> None:
        self.check_matrices = check_matrices
        self.decoder = decoder

    def decode_shots_bit_packed(
        self, bit_packed_detection_event_data: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        obs_flip_data = []
        for shot_data in bit_packed_detection_event_data:
            unpacked_data = np.unpackbits(
                shot_data,
                bitorder="little",
                count=self.check_matrices.check_map.shape[1],
            )
            pred_errors = self.decoder.decode(self.check_matrices.check_map @ unpacked_data)
            obs_pred = (self.check_matrices.obs_matrix @ pred_errors) % 2
            obs_flip_data.append(np.packbits(obs_pred.astype(np.uint8), bitorder="little"))

        return np.array(obs_flip_data)


class BPTypeDecoder(Decoder):
    """
    Base class for running sinter experiments using BP, BP_OSD, and BP_LSD from the ldpc decoding package.
    See provided BP, BPOSD, and BPLSD subclasses.

    NOTE: Currently assumes a CSS code experiment in a specified basis and therefore separates decoding into the resulting X or Z sub-problem.
    """

    def __init__(
        self,
        decoder_cls: BpDecoderClassType,
        basis: PauliXZ,
        fn_det_basis: Callable[[DemTargetWithCoords], PauliXZ],
        **kwargs: Any,
    ) -> None:
        """
        args:
            decoder_cls: BpDecoderClassType
                ldpc package decoder class
            basis: PauliXZ
                CSS decoding sub-problem basis
            fn_det_basis: Callable[[DemTargetWithCoords], PauliXZ]
                For general CSS codes, determining a detector's basis is seemingly non-trivial.
                This function extracts this from the detector coordinates, assuming some convention.
                The default convention is based on first coordinate (1 = Z, 2 = X)
                NOTE: This function needs to be defined at the top-level of a module (i.e. not in a jupyter notebook cell) to work with Sinter
            kwargs: Any
                keyword arguments to be passed to the decoder

        """
        self.decoder_cls = decoder_cls
        self.basis = basis
        self.fn_det_basis = fn_det_basis
        self.decoder_kwargs = kwargs

    def compile_decoder_for_dem(self, dem: DetectorErrorModel) -> CompiledDecoder:
        z_check_matrices, x_check_matrices = detector_error_model_to_css_checks(
            dem, self.fn_det_basis
        )
        if self.basis == Pauli.Z:
            check_matrices = z_check_matrices
        elif self.basis == Pauli.X:
            check_matrices = x_check_matrices
        else:
            raise ValueError(f"Invalid basis: {self.basis}")

        decoder = self.decoder_cls(
            check_matrices.check_matrix,
            error_channel=list(check_matrices.priors),
            **self.decoder_kwargs,
        )
        return CompiledBPTypeDecoder(check_matrices, decoder)

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: pathlib.Path,
        dets_b8_in_path: pathlib.Path,
        obs_predictions_b8_out_path: pathlib.Path,
        tmp_dir: pathlib.Path,
    ) -> None:
        raise NotImplementedError()


class BP(BPTypeDecoder):
    def __init__(
        self,
        basis: PauliXZ,
        fn_det_basis: Callable[[DemTargetWithCoords], PauliXZ] = _det_basis_coord,
        **kwargs: Any,
    ) -> None:
        super().__init__(BpDecoder, basis, fn_det_basis, **kwargs)


class BPOSD(BPTypeDecoder):
    def __init__(
        self,
        basis: PauliXZ,
        fn_det_basis: Callable[[DemTargetWithCoords], PauliXZ] = _det_basis_coord,
        **kwargs: Any,
    ) -> None:
        super().__init__(BpOsdDecoder, basis, fn_det_basis, **kwargs)


class BPLSD(BPTypeDecoder):
    def __init__(
        self,
        basis: PauliXZ,
        fn_det_basis: Callable[[DemTargetWithCoords], PauliXZ] = _det_basis_coord,
        **kwargs: Any,
    ) -> None:
        super().__init__(BpLsdDecoder, basis, fn_det_basis, **kwargs)
