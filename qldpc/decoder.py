"""Decoding a syndrome with a parity check matrix

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from typing import Protocol

import cvxpy
import ldpc
import numpy as np
import numpy.typing as npt
import pymatching


class Decoder(Protocol):
    """Template (protocol) for a decoder object."""

    def __init__(self, matrix: npt.NDArray[np.int_], **decoder_args: object) -> None:
        """Initialize a decoder for the given parity check matrix (and additional arguments)."""

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode the given syndrome."""


def decode(
    matrix: npt.NDArray[np.int_],
    syndrome: npt.NDArray[np.int_],
    **decoder_args: object,
) -> npt.NDArray[np.int_]:
    """Find a `vector` that solves `matrix @ vector == syndrome mod 2`.

    - If passed an explicit decoder, use it.
    - If passed `with_ILP=True`, solve exactly with an integer linear program.
    - If passed `with_MWPM=True`, `with_BF=True`, or `with_BP_OSD=True`, use that decoder.
    - Otherwise, use a BP-LSD decoder.

    In all cases, pass the `decoder_args` to the decoder that is used.
    """
    if callable(custom_decoder := decoder_args.pop("decoder", None)):
        return custom_decoder(matrix, syndrome, **decoder_args)
    return get_decoder(matrix, **decoder_args).decode(syndrome)


def get_decoder(matrix: npt.NDArray[np.int_], **decoder_args: object) -> Decoder:
    """Retrieve a decoder."""
    if decoder_args.pop("with_ILP", False):
        return get_decoder_ILP(matrix, **decoder_args)

    if decoder_args.pop("with_MWPM", False):
        return get_decoder_MWPM(matrix, **decoder_args)

    if decoder_args.pop("with_BF", False):
        return get_decoder_BF(matrix, **decoder_args)

    if decoder_args.pop("with_BP_LSD", False):
        return get_decoder_BP_LSD(matrix, **decoder_args)

    decoder_args.pop("with_BP_OSD", None)
    return get_decoder_BP_OSD(matrix, **decoder_args)


def get_decoder_BP_OSD(matrix: npt.NDArray[np.int_], **decoder_args: object) -> Decoder:
    """Decoder based on belief propagation with ordered statistics (BP+OSD).

    For details about the BD-OSD decoder and its arguments, see:
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - Reference: https://arxiv.org/abs/2005.07016
    """
    return ldpc.BpOsdDecoder(
        matrix,
        error_rate=decoder_args.pop("error_rate", 0.0),
        **decoder_args,
    )


def get_decoder_BP_LSD(matrix: npt.NDArray[np.int_], **decoder_args: object) -> Decoder:
    """Decoder based on belief propagation with localized statistics (BP+LSD).

    For details about the BD-LSD decoder and its arguments, see:
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - Reference: https://arxiv.org/abs/2406.18655
    """
    return ldpc.bplsd_decoder.BpLsdDecoder(
        matrix,
        error_rate=decoder_args.pop("error_rate", 0.0),
        **decoder_args,
    )


def get_decoder_BF(matrix: npt.NDArray[np.int_], **decoder_args: object) -> Decoder:
    """Decoder based on belief finding (BF).

    For details about the BF decoder and its arguments, see:
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - References:
      - https://arxiv.org/abs/1709.06218
      - https://arxiv.org/abs/2103.08049
      - https://arxiv.org/abs/2209.01180
    """
    return ldpc.BeliefFindDecoder(
        matrix,
        error_rate=decoder_args.pop("error_rate", 0.0),
        **decoder_args,
    )


def get_decoder_MWPM(matrix: npt.NDArray[np.int_], **decoder_args: object) -> Decoder:
    """Decoder based on minimum weight perfect matching (MWPM)."""
    return pymatching.Matching.from_check_matrix(matrix, **decoder_args)


def get_decoder_ILP(matrix: npt.NDArray[np.int_], **decoder_args: object) -> Decoder:
    """Decoder based on solving an integer linear program (ILP).

    Supports integers modulo q for q > 2 with a "modulus" argument.

    If a "lower_bound_row" argument is provided, treat this linear constraint (by index) as a lower
    bound (>=), rather than an equality (==) constraint.

    All remaining keyword arguments are passed to `cvxpy.Problem.solve`.
    """
    return DecoderILP(matrix, **decoder_args)


class DecoderILP:
    """Decoder based on solving an integer linear program (ILP).

    Supports integers modulo q for q > 2 with a "modulus" argument.

    If a "lower_bound_row" argument is provided, treat this linear constraint (by index) as a lower
    bound (>=), rather than an equality (==) constraint.

    All remaining keyword arguments are passed to `cvxpy.Problem.solve`.
    """

    def __init__(self, matrix: npt.NDArray[np.int_], **decoder_args: object) -> None:
        modulus = decoder_args.pop("modulus", 2)
        if not isinstance(modulus, int) or modulus < 2:
            raise ValueError(
                f"Decoding problems must have modulus >= 2 (provided modulus: {modulus}"
            )

        lower_bound_row = decoder_args.pop("lower_bound_row", None)
        if not (lower_bound_row is None or isinstance(lower_bound_row, int)):
            raise ValueError(f"Lower bound row index must be an integer, not {lower_bound_row}")

        # variables, their constraints, and the objective (minimizing number of nonzero variables)
        constraints = []
        if modulus == 2:
            variables = cvxpy.Variable(matrix.shape[1], boolean=True)
            objective = cvxpy.Minimize(cvxpy.norm(variables, 1))
        else:
            variables = cvxpy.Variable(matrix.shape[1], integer=True)
            nonzero_variable_flags = cvxpy.Variable(matrix.shape[1], boolean=True)
            constraints += [var >= 0 for var in iter(variables)]
            constraints += [var <= modulus - 1 for var in iter(variables)]
            constraints += [modulus * nonzero_variable_flags >= variables]

            objective = cvxpy.Minimize(cvxpy.norm(nonzero_variable_flags, 1))

        self.matrix = matrix
        self.variables = variables

        self.objective = objective
        self.base_constraints = constraints

        self.modulus = modulus
        self.lower_bound_row = lower_bound_row

        self.decoder_args = decoder_args

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        # constraints for the decoding problem: matrix @ solution == syndrome (mod q)
        constraints = self.base_constraints + _build_cvxpy_constraints(
            self.variables, self.matrix, syndrome, self.modulus, self.lower_bound_row
        )

        # solve the optimization problem!
        problem = cvxpy.Problem(self.objective, constraints)
        result = problem.solve(**self.decoder_args)

        # raise error if the optimization failed
        if not isinstance(result, float) or not np.isfinite(result) or self.variables.value is None:
            message = "Optimal solution to integer linear program could not be found!"
            raise ValueError(message + f"\nSolver output: {result}")

        # return solution to the problem variables
        return self.variables.value.astype(int)


def _build_cvxpy_constraints(
    variables: cvxpy.Variable,
    matrix: npt.NDArray[np.int_],
    syndrome: npt.NDArray[np.int_],
    modulus: int,
    lower_bound_row: int | None = None,
) -> list[cvxpy.Constraint]:
    """Build cvxpy constraints of the form `matrix @ variables == syndrome (mod q)`.

    This method uses boolean slack variables {s_j} to relax each constraint of the form
    `expression = val mod q`
    to
    `expression = val + sum_j q^j s_j`.

    If `lower_bound_row is not None`, treat the constraint at this row index as a lower bound.
    """
    matrix = np.array(matrix) % modulus
    syndrome = np.array(syndrome) % modulus
    if lower_bound_row is not None:
        lower_bound_row %= syndrome.size

    constraints = []
    for idx, (check, syndrome_bit) in enumerate(zip(matrix, syndrome)):
        # identify the largest power of q needed for the relaxation
        max_zero = int(sum(check) * (modulus - 1) - syndrome_bit)
        if max_zero == 0 or modulus == 2:
            max_power_of_q = max_zero.bit_length() - 1
        else:
            max_power_of_q = int(np.log2(max_zero) / np.log2(modulus))

        if max_power_of_q > 0:
            powers_of_q = [modulus**jj for jj in range(1, max_power_of_q + 1)]
            slack_variables = cvxpy.Variable(max_power_of_q, boolean=True)
            zero_mod_q = powers_of_q @ slack_variables
        else:
            zero_mod_q = 0

        if idx == lower_bound_row:
            constraint = check @ variables >= syndrome_bit + zero_mod_q
        else:
            constraint = check @ variables == syndrome_bit + zero_mod_q
        constraints.append(constraint)

    return constraints
