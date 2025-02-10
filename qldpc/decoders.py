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

import itertools
from typing import Callable, Protocol

import cvxpy
import galois
import ldpc
import numpy as np
import numpy.typing as npt
import pymatching

from qldpc import codes


class Decoder(Protocol):
    """Template (protocol) for a decoder object."""

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""


def decode(
    matrix: npt.NDArray[np.int_], syndrome: npt.NDArray[np.int_], **decoder_args: object
) -> npt.NDArray[np.int_]:
    """Find a `vector` that solves `matrix @ vector == syndrome mod 2`."""
    decoder = get_decoder(matrix, **decoder_args)
    return decoder.decode(syndrome)


def get_decoder(matrix: npt.NDArray[np.int_], **decoder_args: object) -> Decoder:
    """Retrieve a decoder."""
    if constructor := decoder_args.pop("decoder_constructor", None):
        assert callable(constructor)
        return constructor(matrix, **decoder_args)

    if decoder := decoder_args.pop("static_decoder", None):
        assert hasattr(decoder, "decode") and callable(getattr(decoder, "decode"))
        assert not decoder_args, "if passed a static decoder, we cannot process decoding arguments"
        return decoder

    if decoder_args.pop("with_lookup", False):
        return get_decoder_lookup(matrix, **decoder_args)

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


def get_decoder_lookup(matrix: npt.NDArray[np.int_], **decoder_args: object) -> LookupDecoder:
    """Decoder based on a lookup table from errors to syndromes."""
    return LookupDecoder(matrix, **decoder_args)  # type:ignore[arg-type]


class LookupDecoder(Decoder):
    """Decoder based on a lookup table from errors to syndromes.

    In addition to a parity check matrix, this decoder can be initialized with a max_weight, in
    which case it builds a lookup table for all errors with weight <= max_weight.  If no max_weight
    is provided, this code computes the distance of a classical code with the provided parity check
    matrix, and sets max_weight to the highest weight for which an error is guaranteed to be
    correctable, namely (code_distance - 1) // 2.
    """

    def __init__(self, matrix: npt.NDArray[np.int_], *, max_weight: int | None = None) -> None:
        field = type(matrix) if isinstance(matrix, galois.FieldArray) else galois.GF(2)
        matrix = field(matrix)

        if max_weight is None:
            code_distance = codes.ClassicalCode(matrix).get_distance()
            max_weight = (code_distance - 1) // 2 if isinstance(code_distance, int) else 0

        self.table: dict[tuple[int, ...], npt.NDArray[np.int_]] = {}
        num_bits = matrix.shape[1]
        for weight in range(max_weight, 0, -1):
            for error_sites in itertools.combinations(range(num_bits), weight):
                error_site_indices = np.asarray(error_sites, dtype=int)
                for errors in itertools.product(range(1, field.order), repeat=weight):
                    code_error = field.Zeros(num_bits)
                    code_error[error_site_indices] = errors
                    syndrome = matrix @ code_error
                    syndrome_bits = tuple(np.where(syndrome)[0])
                    self.table[syndrome_bits] = code_error.view(np.ndarray)

        self.null_correction = np.zeros(num_bits, dtype=int)

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""
        syndrome_bits = tuple(np.where(syndrome)[0])
        return self.table.get(syndrome_bits, self.null_correction.copy())


def get_decoder_ILP(matrix: npt.NDArray[np.int_], **decoder_args: object) -> ILPDecoder:
    """Decoder based on solving an integer linear program (ILP).

    Supports integers modulo q for q > 2 with a "modulus" argument, otherwise uses q = 2.

    If a "lower_bound_row" argument is provided, treat this linear constraint (by index) as a lower
    bound (>=), rather than an equality (==) constraint.

    All remaining keyword arguments are passed to `cvxpy.Problem.solve`.
    """
    return ILPDecoder(matrix, **decoder_args)


class ILPDecoder(Decoder):
    """Decoder based on solving an integer linear program (ILP).

    Supports integers modulo q for q > 2 with a "modulus" argument, otherwise uses q = 2.

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
        assert isinstance(modulus, int)
        self.modulus = modulus

        self.matrix = np.asarray(matrix, dtype=int) % self.modulus
        num_checks, num_variables = self.matrix.shape

        lower_bound_row = decoder_args.pop("lower_bound_row", None)
        if not (lower_bound_row is None or isinstance(lower_bound_row, int)):
            raise ValueError(f"Lower bound row index must be an integer, not {lower_bound_row}")
        assert lower_bound_row is None or isinstance(lower_bound_row, int)
        if isinstance(lower_bound_row, int):
            lower_bound_row %= num_checks
        self.lower_bound_row = lower_bound_row

        # variables, their constraints, and the objective (minimizing number of nonzero variables)
        self.variable_constraints = []
        if self.modulus == 2:
            self.variables = cvxpy.Variable(num_variables, boolean=True)
            self.objective = cvxpy.Minimize(cvxpy.norm(self.variables, 1))
        else:
            self.variables = cvxpy.Variable(num_variables, integer=True)
            nonzero_variable_flags = cvxpy.Variable(num_variables, boolean=True)
            self.variable_constraints += [var >= 0 for var in iter(self.variables)]
            self.variable_constraints += [var <= self.modulus - 1 for var in iter(self.variables)]
            self.variable_constraints += [self.modulus * nonzero_variable_flags >= self.variables]
            self.objective = cvxpy.Minimize(cvxpy.norm(nonzero_variable_flags, 1))

        self.decoder_args = decoder_args

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""
        # identify all constraints
        constraints = self.variable_constraints + self.cvxpy_constraints_for_syndrome(syndrome)

        # solve the optimization problem!
        problem = cvxpy.Problem(self.objective, constraints)
        result = problem.solve(**self.decoder_args)

        # raise error if the optimization failed
        if not isinstance(result, float) or not np.isfinite(result) or self.variables.value is None:
            message = "Optimal solution to integer linear program could not be found!"
            raise ValueError(message + f"\nSolver output: {result}")

        # return solution to the problem variables
        return self.variables.value.astype(int)

    def cvxpy_constraints_for_syndrome(
        self, syndrome: npt.NDArray[np.int_]
    ) -> list[cvxpy.Constraint]:
        """Build cvxpy constraints of the form `matrix @ variables == syndrome (mod q)`.

        This method uses boolean slack variables {s_j} to relax each constraint of the form
        `expression = val mod q`
        to
        `expression = val + sum_j q^j s_j`.

        If `lower_bound_row is not None`, treat the constraint at this row index as a lower bound.
        """
        syndrome = np.array(syndrome) % self.modulus

        constraints = []
        for idx, (check, syndrome_bit) in enumerate(zip(self.matrix, syndrome)):
            # identify the largest power of q needed for the relaxation
            max_zero = int(sum(check) * (self.modulus - 1) - syndrome_bit)
            if max_zero == 0 or self.modulus == 2:
                max_power_of_q = max_zero.bit_length() - 1
            else:
                max_power_of_q = int(np.log2(max_zero) / np.log2(self.modulus))

            if max_power_of_q > 0:
                powers_of_q = [self.modulus**jj for jj in range(1, max_power_of_q + 1)]
                slack_variables = cvxpy.Variable(max_power_of_q, boolean=True)
                zero_mod_q = powers_of_q @ slack_variables
            else:
                zero_mod_q = 0

            if idx == self.lower_bound_row:
                constraint = check @ self.variables >= syndrome_bit + zero_mod_q
            else:
                constraint = check @ self.variables == syndrome_bit + zero_mod_q
            constraints.append(constraint)

        return constraints


class BlockDecoder(Decoder):
    """Decoder for a composite syndrome built from independent identical code blocks.

    A BlockDecoder is instantiated from:
    - the length of a syndrome vector for one code block (syndrome_length), and
    - a decoder for a one code block.
    When asked to decode a syndrome, a BlockDecdoer breaks the syndrome into sections of size
    syndrome_length, and decodes each section using the single-code-block decoder that it was
    instantiated with.
    """

    def __init__(self, syndrome_length: int, decoder: Decoder) -> None:
        self.syndrome_length = syndrome_length
        self.decoder = decoder

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome by parts and return a net inferred error."""
        corrections = [
            self.decoder.decode(sub_syndrome)
            for sub_syndrome in syndrome.reshape(-1, self.syndrome_length)
        ]
        return np.concatenate(corrections)


class DirectDecoder(Decoder):
    """Decoder that maps corrupted code words to corrected code words of a classical code.

    In contrast, an "indirect" decoder maps a syndrome to an error.

    A DirectDecoder can be instantiated from:
    - an indirect decoder, and
    - a parity check matrix.
    When asked to decode a candidate code word, a DirectDecoder first computes a syndrome, decodes
    the syndrome with an indirect decoder to infer an error, and then subtracts the error from the
    candidate word.
    """

    def __init__(self, decode_func: Callable[[npt.NDArray[np.int_]], npt.NDArray[np.int_]]) -> None:
        self.decode_func = decode_func

    def decode(self, word: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a corrupted code word and return an inferred code word."""
        return self.decode_func(word)

    @staticmethod
    def from_indirect(decoder: Decoder, matrix: npt.NDArray[np.int_]) -> DirectDecoder:
        """Instantiate a DirectDecoder from an indirect decoder and a parity check matrix."""
        field = type(matrix) if isinstance(matrix, galois.FieldArray) else None

        if field is None:

            def decode_func(candidate_word: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
                syndrome = matrix @ candidate_word % 2
                return (candidate_word - decoder.decode(syndrome)) % 2

        else:

            def decode_func(candidate_word: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
                candidate_word = field(candidate_word)
                syndrome = matrix @ candidate_word
                error = field(decoder.decode(syndrome.view(np.ndarray)))
                return (candidate_word - error).view(np.ndarray)

        return DirectDecoder(decode_func)
