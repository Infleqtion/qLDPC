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
import warnings
from typing import Callable, Protocol

import cvxpy
import galois
import ldpc
import numpy as np
import numpy.typing as npt
import pymatching

from qldpc import codes
from qldpc.objects import Node, symplectic_conjugate

PLACEHOLDER_ERROR_RATE = 1e-3  # required for some decoding methods


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

    if decoder_args.pop("with_GUF", False):
        return get_decoder_GUF(matrix, **decoder_args)

    if decoder_args.pop("with_ILP", False):
        return get_decoder_ILP(matrix, **decoder_args)

    if decoder_args.pop("with_MWPM", False):
        return get_decoder_MWPM(matrix, **decoder_args)

    if decoder_args.pop("with_BF", False):
        return get_decoder_BF(matrix, **decoder_args)

    if decoder_args.pop("with_BP_LSD", False):
        return get_decoder_BP_LSD(matrix, **decoder_args)

    # use a different default decoder for non-binary fields
    if isinstance(matrix, galois.FieldArray) and type(matrix).order != 2:
        decoder_args.pop("with_GUF", None)
        return get_decoder_GUF(matrix, **decoder_args)

    decoder_args.pop("with_BP_OSD", None)
    return get_decoder_BP_OSD(matrix, **decoder_args)


def get_decoder_BP_OSD(matrix: npt.NDArray[np.int_], **decoder_args: object) -> Decoder:
    """Decoder based on belief propagation with ordered statistics (BP+OSD).

    For details about the BD-OSD decoder and its arguments, see:
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - Reference: https://arxiv.org/abs/2005.07016
    """
    if "error_channel" not in decoder_args and "error_rate" not in decoder_args:
        decoder_args["error_rate"] = PLACEHOLDER_ERROR_RATE
    return ldpc.BpOsdDecoder(matrix, **decoder_args)


def get_decoder_BP_LSD(matrix: npt.NDArray[np.int_], **decoder_args: object) -> Decoder:
    """Decoder based on belief propagation with localized statistics (BP+LSD).

    For details about the BD-LSD decoder and its arguments, see:
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - Reference: https://arxiv.org/abs/2406.18655
    """
    if "error_channel" not in decoder_args and "error_rate" not in decoder_args:
        decoder_args["error_rate"] = PLACEHOLDER_ERROR_RATE
    return ldpc.bplsd_decoder.BpLsdDecoder(matrix, **decoder_args)


def get_decoder_BF(matrix: npt.NDArray[np.int_], **decoder_args: object) -> Decoder:
    """Decoder based on belief finding (BF).

    For details about the BF decoder and its arguments, see:
    - Documentation: https://software.roffe.eu/ldpc/quantum_decoder.html
    - References:
      - https://arxiv.org/abs/1709.06218
      - https://arxiv.org/abs/2103.08049
      - https://arxiv.org/abs/2209.01180
    """
    if "error_channel" not in decoder_args and "error_rate" not in decoder_args:
        decoder_args["error_rate"] = PLACEHOLDER_ERROR_RATE
    return ldpc.BeliefFindDecoder(matrix, **decoder_args)


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

    If initialized with symplectic=True, this decoder treats the provided parity check matrix as that
    of a QuditCode, with the first and last half of the columns denoting, respectively, the X and Z
    support of a stabilizer.  Decoded errors are likewise vectors that indicate their X and Z
    support by the first and second half of their entries.
    """

    def __init__(
        self,
        matrix: npt.NDArray[np.int_],
        *,
        max_weight: int | None = None,
        symplectic: bool = False,
    ) -> None:
        code: codes.ClassicalCode | codes.QuditCode
        if not symplectic:
            code = codes.ClassicalCode(matrix)
            field = code.field
            matrix = code.matrix
        else:
            code = codes.QuditCode(matrix)
            field = code.field
            matrix = symplectic_conjugate(code.matrix)

        if max_weight is None:
            warnings.warn(
                "A LookupDecoder has been initialized without specifying a maximum error weight,"
                " max_weight, so max_weight is being set to (code_distance - 1) // 2, where"
                " code_distance is the distance of the ClassicalCode (if symplectic is False) or"
                " QuditCode (if symplectic is True) with the parity check matrix provided to the"
                " LookupDecoder.  This default choice of max_weight is a poor choice for decoding"
                " in one sector of a qudit CSSCode, for which the LookupDecoder does not have"
                " enough information to automatically determine a reasonable max_weight."
            )
            code_distance = code.get_distance()
            max_weight = (code_distance - 1) // 2 if isinstance(code_distance, int) else 0

        # identify the set of local errors that can occur
        repeat = 2 if symplectic else 1
        local_errors = tuple(itertools.product(range(field.order), repeat=repeat))[1:]

        self.table: dict[tuple[int, ...], npt.NDArray[np.int_]] = {}
        block_length = matrix.shape[1] // repeat
        for weight in range(max_weight, 0, -1):
            for error_sites in itertools.combinations(range(block_length), weight):
                error_site_indices = list(error_sites)
                for errors in itertools.product(local_errors, repeat=weight):
                    code_error = field.Zeros((repeat, block_length))
                    code_error[:, error_site_indices] = np.asarray(errors, dtype=int).T
                    code_error = code_error.ravel()
                    syndrome = matrix @ code_error
                    self.table[tuple(syndrome.view(np.ndarray))] = code_error.view(np.ndarray)

        self.null_correction = np.zeros(matrix.shape[1], dtype=int)

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""
        return self.table.get(tuple(syndrome.view(np.ndarray)), self.null_correction.copy())


def get_symplectic_weight(vector: npt.NDArray[np.int_]) -> int:
    """Get the symplectic weight of a vector.

    The symplectic weight of a Pauli string is the number of qudits that it addresses nontrivially.
    """
    vector = vector.reshape(2, -1)
    vector_x = np.asarray(vector[0], dtype=int)
    vector_z = np.asarray(vector[1], dtype=int)
    return np.count_nonzero(vector_x | vector_z)


def get_decoder_GUF(matrix: npt.NDArray[np.int_], **decoder_args: object) -> GUFDecoder:
    """Decoder based on a generalization of Union-Find, described in arXiv:2103.08049."""
    return GUFDecoder(matrix, **decoder_args)  # type:ignore[arg-type]


class GUFDecoder(Decoder):
    """The generalized Union-Find (GUF) decoder in https://arxiv.org/abs/2103.08049.

    If passed a max_weight argument, this decoder tries to find an error with weight <= max_weight,
    and returns the first such error that it finds.  If no such error is found, this decoder returns
    the minimum-weight error that it found while trying.  Be warned that passing a max_weight makes
    this decoder have worst-case exponential runtime.

    If initialized with symplectic=True, this decoder treats the provided parity check matrix as that
    of a QuditCode, with the first and last half of the columns denoting, respectively, the X and Z
    support of a stabilizer.  Decoded errors are likewise vectors that indicate their X and Z
    support by the first and second half of their entries.

    Warning: this implementation of the generalized Union-Find decoder is highly unoptimized.  For
    one, it is written entirely in Python.  Moreover, this implementation does not factor an error
    set into connected componenents.
    """

    def __init__(
        self,
        matrix: npt.NDArray[np.int_],
        *,
        max_weight: int | None = None,
        symplectic: bool = False,
    ) -> None:
        self.default_max_weight = max_weight
        self.symplectic = symplectic

        self.get_weight: Callable[[npt.NDArray[np.int_]], int]
        self.code: codes.AbstractCode
        if not symplectic:
            # "ordinary" decoding of a classical code
            self.get_weight = np.count_nonzero  # Hamming weight (of an error vector)
            self.code = codes.ClassicalCode(matrix)

        else:
            # decoding a quantum code: the "weight" of an error vector is its symplectic weight
            self.get_weight = get_symplectic_weight
            self.code = codes.QuditCode(symplectic_conjugate(matrix))

        self.graph = self.code.graph.to_undirected()

    def decode(
        self, syndrome: npt.NDArray[np.int_], *, max_weight: int | None = None
    ) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""
        max_weight = max_weight if max_weight is not None else self.default_max_weight
        syndrome = self.code.field(syndrome)
        syndrome_bits = np.where(syndrome)[0]

        # construct an "error set", within which we look for solutions to the decoding problem
        error_set = set(Node(index, is_data=False) for index in syndrome_bits)
        solutions = np.zeros((0, len(self.code)), dtype=int)
        last_error_set_size = 0
        while solutions.size == 0:
            # grow the error set by one step on the Tanner graph
            error_set |= set(
                neighbor for node in error_set for neighbor in self.graph.neighbors(node)
            )

            # if the error set has not grown, there is no valid solution, so exit now
            if len(error_set) == last_error_set_size:
                return np.zeros(len(self.code) * (2 if self.symplectic else 1), dtype=int)
            last_error_set_size = len(error_set)

            # check whether the syndrome can be induced by errors in the interior of the error_set
            checks, bits = self.get_sub_problem_indices(syndrome, error_set)
            sub_matrix = self.code.matrix[np.ix_(checks, bits)]
            sub_syndrome = syndrome[checks]

            """
            Try to identify errors in the interior of the error_set that reproduce the syndrome,
            looking for solutions x to H @ x = s, or solutions [y,c] to [H|-s] @ [y,c].T = 0.
            """
            augmented_matrix = np.column_stack([sub_matrix, -sub_syndrome])
            candidate_solutions = augmented_matrix.null_space()  # type:ignore[attr-defined]
            solutions = candidate_solutions[np.where(candidate_solutions[:, -1])]

        # convert solutions [y,c] --> [y/c,1] --> y
        if self.code.field.order == 2:
            solutions = solutions[:, :-1]
        else:
            solutions = solutions[:, :-1] / solutions[:, -1][:, None]

        # identify the minimum-weight solution found so far
        min_weight_solution = min(solutions, key=lambda solution: self.get_weight(solution))
        weight = self.get_weight(min_weight_solution)

        if max_weight is not None and weight > max_weight:
            # identify null-syndrome vectors
            null_vectors = sub_matrix.null_space()

            # minimize the weight of the solution over additions of null-syndrome vectors
            min_weight = weight
            one_solution = min_weight_solution.copy()
            null_vector_coefficients = itertools.product(
                self.code.field.elements, repeat=len(null_vectors)
            )
            next(null_vector_coefficients)  # skip the all-0 vector of coefficients
            for coefficients in null_vector_coefficients:
                solution = one_solution + self.code.field(coefficients) @ null_vectors
                weight = self.get_weight(solution)
                if weight < min_weight:
                    min_weight = weight
                    min_weight_solution = solution
                    if weight <= max_weight:
                        break

        # construct the full error
        error = self.code.field.Zeros(len(self.code) * (2 if self.symplectic else 1))
        error[bits] = min_weight_solution
        return error.view(np.ndarray)

    def get_sub_problem_indices(
        self, syndrome: npt.NDArray[np.int_], error_set: set[Node]
    ) -> tuple[list[int], list[int]]:
        """Syndrome and data bit indices for decoding on the interior of the given error set."""
        # identify the "interior" of error set: nodes whose neighbors are contained in the set
        interior_nodes = [
            node for node in error_set if error_set.issuperset(self.graph.neighbors(node))
        ]
        # identify interior data bit nodes, and their neighbors
        interior_data_nodes = [node for node in interior_nodes if node.is_data]
        check_nodes = set(node for node in error_set if not node.is_data) | set(
            neighbor for node in interior_data_nodes for neighbor in self.graph.neighbors(node)
        )
        checks = [node.index for node in check_nodes]
        bits = [node.index for node in interior_data_nodes]

        if self.symplectic:
            # add classical bits to account for the support of Z-type operators in the error vector
            bits += [bit + len(self.code) for bit in bits]

        # the order of checks, bits is technically arbitrary, but according to unofficial empirical
        # tests, reverse-sorted order works better for concatenated codes
        return sorted(checks, reverse=True), sorted(bits, reverse=True)


def get_decoder_ILP(matrix: npt.NDArray[np.int_], **decoder_args: object) -> ILPDecoder:
    """Decoder based on solving an integer linear program (ILP).

    All remaining keyword arguments are passed to `cvxpy.Problem.solve`.
    """
    return ILPDecoder(matrix, **decoder_args)


class ILPDecoder(Decoder):
    """Decoder based on solving an integer linear program (ILP).

    All remaining keyword arguments are passed to `cvxpy.Problem.solve`.
    """

    def __init__(self, matrix: npt.NDArray[np.int_], **decoder_args: object) -> None:
        self.modulus = type(matrix).order if isinstance(matrix, galois.FieldArray) else 2
        if not galois.is_prime(self.modulus):
            raise ValueError("ILP decoding only supports prime number fields")

        self.matrix = np.asarray(matrix, dtype=int) % self.modulus
        num_checks, num_variables = self.matrix.shape

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
        """
        syndrome = np.asarray(syndrome, dtype=int) % self.modulus

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
    """Decoder that maps corrupted code words to corrected code words.

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
