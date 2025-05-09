from __future__ import annotations

from collections.abc import Callable
import pathlib
import numpy as np
from sinter import CompiledDecoder, Decoder
from stim import DetectorErrorModel, DemTargetWithCoords
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.bp_decoder import BpDecoder, BpDecoderBase
from ldpc.bplsd_decoder import BpLsdDecoder

from qldpc.stim.util import (
    _det_basis_coord,
    detector_error_model_to_css_checks,
    CheckMatrices,
    Basis,
)


class CompiledBPTypeDecoder(CompiledDecoder):

    def __init__(self, check_matrices: CheckMatrices, decoder):
        self.check_matrices = check_matrices
        self.decoder = decoder

    def decode_shots_bit_packed(
        self, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        obs_flip_data = []
        for shot_data in bit_packed_detection_event_data:
            unpacked_data = np.unpackbits(
                shot_data,
                bitorder="little",
                count=self.check_matrices.check_map.shape[1],
            )
            pred_errors = self.decoder.decode(
                self.check_matrices.check_map @ unpacked_data
            )
            obs_pred = (self.check_matrices.obs_matrix @ pred_errors) % 2
            obs_flip_data.append(
                np.packbits(obs_pred.astype(np.uint8), bitorder="little")
            )

        return np.array(obs_flip_data)


class BPTypeDecoder(Decoder):

    def __init__(
        self,
        decoder_cls: BpDecoderBase,
        basis: Basis,
        fn_det_basis: Callable[[DemTargetWithCoords], Basis] = _det_basis_coord,
        **kwargs,
    ):
        self.decoder_cls = decoder_cls
        self.basis = basis
        self.fn_det_basis = fn_det_basis
        self.decoder_kwargs = kwargs

    def compile_decoder_for_dem(self, dem: DetectorErrorModel) -> CompiledDecoder:
        z_check_matrices, x_check_matrices = detector_error_model_to_css_checks(
            dem, self.fn_det_basis
        )
        if self.basis == Basis.Z:
            check_matrices = z_check_matrices
        elif self.basis == Basis.X:
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

    def __init__(self, basis: Basis, **kwargs):
        super().__init__(BpDecoder, basis, **kwargs)


class BPOSD(BPTypeDecoder):

    def __init__(self, basis: Basis, **kwargs):
        super().__init__(BpOsdDecoder, basis, **kwargs)


class BPLSD(BPTypeDecoder):

    def __init__(self, basis: Basis, **kwargs):
        super().__init__(BpLsdDecoder, basis, **kwargs)
