"""General error-correcting code classes and methods

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

import abc
import functools
import itertools
import random
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Iterator, Literal, TypeVar, cast

import galois
import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.special
import stim

from qldpc import abstract, decoders, external
from qldpc.abstract import DEFAULT_FIELD_ORDER
from qldpc.objects import (
    PAULIS_XZ,
    Node,
    Pauli,
    PauliXZ,
    QuditOperator,
    op_to_string,
    symplectic_conjugate,
)

from ._distance import get_distance_classical, get_distance_quantum

IntegerArray = TypeVar("IntegerArray", npt.NDArray[np.int_], galois.FieldArray)
Slice = slice | npt.NDArray[np.int_] | list[int]


def get_scrambled_seed(seed: int) -> int:
    """Scramble a seed, allowing us to safely increment seeds in repeat-until-success protocols."""
    state = np.random.get_state()
    np.random.seed(seed)
    new_seed = np.random.randint(np.iinfo(np.int32).max + 1)
    np.random.set_state(state)
    return new_seed


def get_random_array(
    field: type[galois.FieldArray],
    shape: int | tuple[int, ...],
    *,
    satisfy: Callable[[galois.FieldArray], bool | np.bool_] = lambda _: True,
    seed: int | None = None,
) -> galois.FieldArray:
    """Get a random array over a given finite field with a given shape.

    If passed a condition that the array must satisfy, re-sample until the condition is met.
    """
    seed = get_scrambled_seed(seed) if seed is not None else None
    while not satisfy(array := field.Random(shape, seed=seed)):
        seed = seed + 1 if seed is not None else None  # pragma: no cover
    return array


################################################################################
# template error-correcting code class


class AbstractCode(abc.ABC):
    """Template class for error-correcting codes."""

    _field: type[galois.FieldArray]
    _exact_distance: int | float | None = None

    def __init__(
        self,
        matrix: AbstractCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | None = None,
    ) -> None:
        """Construct a code from a parity check matrix over a finite field.

        The base field is taken to be F_2 by default.
        """
        self._matrix: galois.FieldArray
        if isinstance(matrix, AbstractCode):
            self._field = matrix.field
            self._matrix = matrix.matrix
        elif isinstance(matrix, galois.FieldArray):
            self._field = type(matrix)
            self._matrix = matrix
        else:
            self._field = galois.GF(field or DEFAULT_FIELD_ORDER)
            self._matrix = self._field(matrix)

        if field is not None and field != self._field.order:
            raise ValueError(
                f"Field argument {field} is inconsistent with the given code, which is defined"
                f" over F_{self.field.order}"
            )

    @property
    def name(self) -> str:
        """The name of this code."""
        return getattr(self, "_name", type(self).__name__)

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field over which this code is defined."""
        return self._field

    @property
    def field_name(self) -> str:
        """The name of the base field of this code."""
        characteristic = self.field.characteristic
        degree = self.field.degree
        order = str(characteristic) + (f"^{degree}" if degree > 1 else "")
        return f"GF({order})"

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        """Equality test between two code instances."""

    @abc.abstractmethod
    def __str__(self) -> str:
        """Human-readable representation of this code."""

    @property
    def matrix(self) -> galois.FieldArray:
        """Parity check matrix of this code."""
        return self._matrix

    @functools.cached_property
    @abc.abstractmethod
    def canonicalized(self) -> AbstractCode:
        """The same code with its parity matrix in reduced row echelon form."""

    @staticmethod
    def equiv(code_a: AbstractCode, code_b: AbstractCode) -> bool:
        """Do the two codes have the same parity checks?"""
        return code_a.field is code_b.field and np.array_equal(
            code_a.canonicalized.matrix, code_b.canonicalized.matrix
        )

    @abc.abstractmethod
    def __len__(self) -> int:
        """The block length of this code."""

    @property
    def num_checks(self) -> int:
        """Number of parity checks in this code."""
        return self.matrix.shape[0]

    @functools.cached_property
    def rank(self) -> int:
        """Rank of this code's parity check matrix."""
        return np.count_nonzero(np.any(self.canonicalized.matrix, axis=1))

    @functools.cached_property
    def graph(self) -> nx.DiGraph:
        """Tanner graph of this code."""
        return self.matrix_to_graph(self.matrix)

    @staticmethod
    @abc.abstractmethod
    def matrix_to_graph(matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]]) -> nx.DiGraph:
        """Convert a parity check matrix into a Tanner graph."""

    @staticmethod
    @abc.abstractmethod
    def graph_to_matrix(graph: nx.DiGraph) -> galois.FieldArray:
        """Convert a Tanner graph into a parity check matrix."""

    @functools.cached_property
    @abc.abstractmethod
    def dimension(self) -> int:
        """The number of logical (qu)dits encoded by this code."""


################################################################################
# classical codes


class ClassicalCode(AbstractCode):
    """Classical linear error-correcting code over a finite field F_q.

    A classical binary code C = {x} is a set of vectors x (with entries in F_q) called code words.
    We consider only linear codes, for which any linear combination of code words is also code word.

    Operationally, we define a classical code by a parity check matrix H with dimensions
    (num_checks, num_bits).  Each row of H represents a linear constraint (a "check") that code
    words must satisfy.  A vector x is a code word iff H @ x = 0.
    """

    _matrix: galois.FieldArray

    def __eq__(self, other: object) -> bool:
        """Equality test between two code instances."""
        return (
            isinstance(other, ClassicalCode)
            and self.field is other.field
            and np.array_equal(self.matrix, other.matrix)
        )

    def __str__(self) -> str:
        """Human-readable representation of this code."""
        text = ""
        if self.field.order == 2:
            text += f"{self.name} on {len(self)} bits"
        else:
            text += f"{self.name} on {len(self)} symbols over {self.field_name}"
        text += f", with parity check matrix\n{self.matrix}"
        return text

    def __contains__(
        self, words: npt.NDArray[np.int_] | Sequence[int] | Sequence[Sequence[int]]
    ) -> bool:
        """Does this code contain the given word(s)?"""
        return not np.any(self.matrix @ self.field(words).T)

    @functools.cached_property
    def canonicalized(self) -> ClassicalCode:
        """The same code with its parity matrix in reduced row echelon form."""
        matrix_rref = self.matrix.row_reduce()
        matrix_rref = matrix_rref[np.any(matrix_rref, axis=1), :]
        return ClassicalCode(matrix_rref, self.field.order)

    def __len__(self) -> int:
        """The block length of this code."""
        return self.matrix.shape[1]

    @property
    def num_bits(self) -> int:
        """Number of data bits in this code."""
        return len(self)

    @staticmethod
    def matrix_to_graph(matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]]) -> nx.DiGraph:
        """Convert a parity check matrix into a Tanner graph.

        The Tanner graph is a bipartite graph with (num_checks, num_bits) vertices, respectively
        identified with the checks and bits of the code.  The check vertex c and the bit vertex b
        share an edge iff c addresses b; that is, edge (c, b) is in the graph iff H[c, b] != 0.
        """
        graph = nx.DiGraph()
        for row, col in zip(*np.nonzero(matrix)):
            node_c = Node(index=int(row), is_data=False)
            node_d = Node(index=int(col), is_data=True)
            graph.add_edge(node_c, node_d, val=matrix[row][col])
        if isinstance(matrix, galois.FieldArray):
            graph.order = type(matrix).order
        return graph

    @staticmethod
    def graph_to_matrix(graph: nx.DiGraph) -> galois.FieldArray:
        """Convert a Tanner graph into a parity check matrix."""
        num_bits = sum(node.is_data for node in graph.nodes())
        num_checks = len(graph.nodes()) - num_bits
        field = getattr(graph, "order", DEFAULT_FIELD_ORDER)
        matrix = galois.GF(field).Zeros((num_checks, num_bits))
        for node_c, node_b, data in graph.edges(data=True):
            matrix[node_c.index, node_b.index] = data.get("val", 1)
        return matrix

    def get_weight(self) -> int:
        """Compute the weight of the largest parity check."""
        return np.max(np.count_nonzero(self.matrix.view(np.ndarray), axis=1))

    @functools.cached_property
    def dimension(self) -> int:
        """The number of logical bits encoded by this code."""
        return len(self) - self.rank

    @functools.cached_property
    def generator(self) -> galois.FieldArray:
        """Generator of this code: a matrix whose rows form a basis for all code words."""
        return self.matrix.null_space()

    def iter_words(self, skip_zero: bool = False) -> Iterator[galois.FieldArray]:
        """Iterate over the code words of this code."""
        vectors = itertools.product(self.field.elements, repeat=self.generator.shape[0])
        if skip_zero:
            # skip the all-0 vector
            next(vectors)
        for vector in vectors:
            yield self.field(vector) @ self.generator

    def get_random_word(self, *, seed: int | None = None) -> galois.FieldArray:
        """Random code word: a sum of all generating words with random field coefficients."""
        num_words = self.generator.shape[0]
        return get_random_array(self.field, num_words, seed=seed) @ self.generator

    def dual(self) -> ClassicalCode:
        """Dual to this code.

        The dual code ~C is the set of bitstrings orthogonal to C:
        ~C = { x : x @ y = 0 for all y in C }.
        The parity check matrix of ~C is equal to the generator of C.
        """
        return ClassicalCode(self.generator)

    def __invert__(self) -> ClassicalCode:
        return self.dual()

    @staticmethod
    def tensor_product(code_a: ClassicalCode, code_b: ClassicalCode) -> ClassicalCode:
        """Tensor product C_a ⨂ C_b of two codes C_a and C_b.

        Let G_a and G_b respectively denote the generators C_a and C_b.
        Definition: C_a ⨂ C_b is the code whose generators are G_a ⨂ G_b.

        Observation: G_a ⨂ G_b is the check matrix of ~(C_a ⨂ C_b).
        We therefore construct ~(C_a ⨂ C_b) and return its dual ~~(C_a ⨂ C_b) = C_a ⨂ C_b.
        """
        if code_a.field is not code_b.field:
            raise ValueError("Cannot take tensor product of codes over different fields")
        gen_a: npt.NDArray[np.int_] = code_a.generator
        gen_b: npt.NDArray[np.int_] = code_b.generator
        return ~ClassicalCode(np.kron(gen_a, gen_b))

    def get_distance(
        self,
        *,
        bound: int | bool | None = None,
        vector: Sequence[int] | npt.NDArray[np.int_] | None = None,
        **decoder_args: Any,
    ) -> int | float:
        """Compute (or upper bound) the minimum Hamming weight of nontrivial code words.

        If `bound is None`, compute an exact code distance by brute force.  Otherwise, compute
        upper bounds using a randomized algorithm and minimize over `bound` trials.  For a detailed
        explanation, see `get_one_distance_bound`.

        If provided a vector, compute (or bound) the minimum Hamming distance between this vector
        and a code word.

        Additional arguments, if applicable, are passed to a decoder in `get_one_distance_bound`.
        """
        if not bound:
            return self.get_distance_exact(vector=vector)
        return self.get_distance_bound(num_trials=int(bound), vector=vector, **decoder_args)

    def get_distance_exact(
        self, *, vector: Sequence[int] | npt.NDArray[np.int_] | None = None
    ) -> int | float:
        """Compute the minimum weight of nontrivial code words by brute force.

        If passed a vector, compute the minimum Hamming distance between the vector and a code word.
        """
        if (known_distance := self.get_distance_if_known(vector)) is not None:
            return known_distance

        if vector is not None:
            vector = self.field(vector)
            return min(np.count_nonzero(word - vector) for word in self.iter_words())

        # we do not know the exact distance, so compute it
        if self.field.order == 2:
            distance = get_distance_classical(self.generator.view(np.ndarray).astype(np.uint8))
        else:
            warnings.warn(
                "Computing the exact distance of a non-binary code may take a (very) long time"
            )
            distance = min(np.count_nonzero(word) for word in self.iter_words(skip_zero=True))
        self._exact_distance = int(distance)
        return self._exact_distance

    def get_distance_if_known(
        self, vector: Sequence[int] | npt.NDArray[np.int_] | None = None
    ) -> int | float | None:
        """Retrieve exact distance, if known.  Otherwise return None."""
        if vector is not None:
            return np.count_nonzero(vector) if self.dimension == 0 else None

        # the distance of dimension-0 codes is undefined
        if self.dimension == 0:
            self._exact_distance = np.nan

        return self._exact_distance

    def get_distance_bound(
        self,
        num_trials: int = 1,
        *,
        cutoff: int | None = None,
        vector: Sequence[int] | npt.NDArray[np.int_] | None = None,
        **decoder_args: Any,
    ) -> int | float:
        """Compute an upper bound on code distance by minimizing many individual upper bounds.

        If passed a cutoff, don't bother trying to find distances less than the cutoff.

        If passed a vector, compute the minimum Hamming distance between the vector and a code word.

        Additional arguments, if applicable, are passed to a decoder in `get_one_distance_bound`.
        """
        if (known_distance := self.get_distance_if_known(vector)) is not None:
            return known_distance

        min_bound = len(self)
        for _ in range(num_trials):
            if cutoff and min_bound <= cutoff:
                break
            new_bound = self.get_one_distance_bound(vector=vector, **decoder_args)
            min_bound = int(min(min_bound, new_bound))
        return min_bound

    def get_one_distance_bound(
        self, *, vector: Sequence[int] | npt.NDArray[np.int_] | None = None, **decoder_args: Any
    ) -> int:
        """Use a randomized algorithm to compute a single upper bound on code distance.

        The code distance is the minimum Hamming distance between two code words, or equivalently
        the minimum Hamming weight of a nonzero code word.  To find a minimal nonzero code word we
        decode a trivial (all-0) syndrome, but enforce that the code word has nonzero overlap with a
        random word, which excludes the all-0 word as a candidate.

        If passed a vector, bound the minimum Hamming distance between the vector and a code word.
        Equivalently, we can interpret the given vector as an error, and find a minimal-weight
        correction from decoding the syndrome induced by this vector.

        Additional arguments, if applicable, are passed to a decoder.
        """
        if vector is not None:
            # find the distance of the given vector from a code word
            correction = decoders.decode(
                self.matrix, self.matrix @ self.field(vector), **decoder_args
            )
            return int(np.count_nonzero(correction))

        # effective syndrome: a trivial "actual" syndrome, and a nonzero overlap with a random word
        effective_syndrome = np.zeros(self.num_checks + 1, dtype=int)
        effective_syndrome[-1] = 1

        valid_candidate_found = False
        while not valid_candidate_found:
            # construct an effective check matrix with a random nonzero word
            random_word = get_random_array(self.field, len(self), satisfy=lambda vec: vec.any())
            effective_check_matrix = np.vstack([self.matrix, random_word])

            # find a low-weight candidate code word
            candidate = decoders.decode(effective_check_matrix, effective_syndrome, **decoder_args)

            # check whether we found a valid candidate
            actual_syndrome = effective_check_matrix @ self.field(candidate)
            valid_candidate_found = np.array_equal(actual_syndrome, effective_syndrome)

        return int(np.count_nonzero(candidate))

    def get_code_params(
        self, *, bound: int | bool | None = None, **decoder_args: Any
    ) -> tuple[int, int, int | float]:
        """Compute the parameters of this code: [n,k,d].

        Here:
        - n is the number of data bits
        - k is the number of encoded ("logical") bits
        - d is the code distance

        If `bound is None`, compute an exact code distance by brute force.  Otherwise, compute an
        upper bound using `bound` trials of a a randomized algorithm.  For a detailed explanation,
        see the `get_one_distance_bound` method.

        Keyword arguments are passed to the calculation of code distance.
        """
        distance = self.get_distance(bound=bound, vector=None, **decoder_args)
        return len(self), self.dimension, distance

    @staticmethod
    def random(
        bits: int, checks: int, field: int | None = None, *, seed: int | None = None
    ) -> ClassicalCode:
        """Construct a random linear code with the given number of bits and checks.

        Reject any code with trivial checks or unchecked bits, identified by an all-zero row or
        column in the code's parity check matrix.
        """
        code_field = galois.GF(field or DEFAULT_FIELD_ORDER)

        def nontrivial(matrix: galois.FieldArray) -> bool:
            """Return True iff all rows and columns are nonzero."""
            return all(row.any() for row in matrix) and all(col.any() for col in matrix.T)

        matrix = get_random_array(code_field, (checks, bits), satisfy=nontrivial, seed=seed)
        return ClassicalCode(matrix)

    @staticmethod
    def from_generator(
        generator: npt.NDArray[np.int_] | Sequence[Sequence[int]], field: int | None = None
    ) -> ClassicalCode:
        """Construct a ClassicalCode from a generator matrix."""
        return ~ClassicalCode(generator, field)

    @staticmethod
    def from_name(name: str) -> ClassicalCode:
        """Named code in the GAP computer algebra system."""
        standardized_name = name.strip().replace(" ", "")  # remove whitespace
        matrix, field = external.codes.get_code(standardized_name)
        code = ClassicalCode(matrix, field)
        setattr(code, "_name", name)
        return code

    def get_automorphism_group(self) -> abstract.Group:
        """Get the automorphism group of this code.

        The auomorphism group of a classical linear code is the group of permutations of bits that
        preserve the code space.
        """
        matrix = self.canonicalized.matrix.view(np.ndarray)
        checks_str = ["[" + ",".join(map(str, line)) + "]" for line in matrix]
        matrix_str = "[" + ",".join(checks_str) + "]"
        code_str = f"CheckMatCode({matrix_str}, GF({self.field.order}))"
        group_str = "AutomorphismGroup" if self.field.order == 2 else "PermutationAutomorphismGroup"
        return abstract.Group.from_name(f"{group_str}({code_str})", field=self.field.order)

    @staticmethod
    def stack(*codes: ClassicalCode) -> ClassicalCode:
        """Stack the given classical codes.

        The stacked code is obtained by having the input codes act on disjoint sets of bits.
        Stacking two codes with parameters [n_1, k_1, d_1] and [n_2, k_2, d_2], for example, results
        in a single code with parameters [n_1 + n_2, k_1 + k_2, min(d_1, d_2)].
        """
        fields = [code.field for code in codes]
        if len(set(fields)) > 1:
            raise ValueError("Cannot stack codes over different fields")
        matrices = [code.matrix for code in codes]
        return ClassicalCode(scipy.linalg.block_diag(*matrices), field=fields[0].order)

    def puncture(self, *bits: int) -> ClassicalCode:
        """Delete the specified bits from a code.

        To delete bits from the code, we remove the corresponding columns from its generator matrix
        (whose rows are code words that form a basis for the code space).
        """
        assert all(0 <= bit < len(self) for bit in bits)
        bits_to_keep = [bit for bit in range(len(self)) if bit not in bits]
        new_generator = self.generator[:, bits_to_keep]
        return ClassicalCode.from_generator(new_generator, self.field.order)

    def shorten(self, *bits: int) -> ClassicalCode:
        """Shorten a code to the words that are zero on the specified bits, and delete those bits.

        To shorten a code on a given bit, we:
        - move the bit to the first position,
        - row-reduce the generator matrix into the form [ identity_matrix, other_stuff ], and
        - delete the first row and column from the generator matrix.
        """
        assert all(0 <= bit < len(self) for bit in bits)
        generator = self.generator
        for bit in sorted(bits, reverse=True):
            generator = np.roll(generator, -bit, axis=1)  # type:ignore[assignment]
            generator = generator.row_reduce()[1:, 1:]
            generator = np.roll(generator, bit, axis=1)  # type:ignore[assignment]
        return ClassicalCode.from_generator(generator)

    def get_logical_error_rate_func(
        self, num_samples: int, max_error_rate: float = 0.3, **decoder_args: Any
    ) -> Callable[[float | Sequence[float]], tuple[float, float]]:
        """Construct a function from physical --> logical error rate in a code capacity model.

        In addition to the logical error rate, the constructed function returns an uncertainty
        (standard error) in that logical error rate.

        The physical error rate provided to the constructed function is the probability with which
        each bit experiences a bit-flip error.  The constructed function will throw an error if
        given a physical error rate larger than max_error_rate.

        The logical error rate returned by the constructed function the probability with which a
        code error (obtained by sampling independent errors on all bits) is decoded incorrectly.

        The basic idea in this method is to first think of the decoding fidelity F(p) = 1 -
        logical_error_rate(p) as a function of the physical error rate p, and decompose
            F(p) = sum_k q_k(p) F_k,
        where q_k(p) = (n choose k) p**k (1-p)**(n-k) is the probability of a weight-k error (here n
        is total number of bits in the code), and F_k is the probability with which a weight-k error
        is corrected by the decoder.  Importantly, F_k is independent of p.  We therefore use our
        sample budget to compute estimates of F_k (according to some allocation of samples to each
        weight k, which depends on the max_error_rate), and then recycle the values of F_k to
        compute each F(p).

        There is one more minor trick, which is that we can use the fact that F_k = 1 to simplify
            F(p) = q_0(p) + sum_(k>0) q_k(p) F_k.
        We thereby only need to sample errors of weight k > 0.
        """
        decoder = decoders.get_decoder(self.matrix, **decoder_args)
        if not isinstance(decoder, decoders.DirectDecoder):
            decoder = decoders.DirectDecoder.from_indirect(decoder, self.matrix)

        # compute decoding fidelities for each error weight
        sample_allocation = _get_sample_allocation(num_samples, len(self), max_error_rate)
        max_error_weight = len(sample_allocation) - 1
        fidelities = np.ones(max_error_weight + 1, dtype=float)
        variances = np.zeros(max_error_weight + 1, dtype=float)
        for weight in range(1, max_error_weight + 1):
            fidelities[weight], variances[weight] = self._estimate_decoding_fidelity_and_variance(
                weight, sample_allocation[weight], decoder
            )

        @np.vectorize
        def get_logical_error_rate(error_rate: float) -> tuple[float, float]:
            """Compute a logical error rate in a code-capacity model."""
            if error_rate > max_error_rate:
                raise ValueError(
                    "Cannot determine logical error rates for physical error rates greater than"
                    f" {max_error_rate}.  Try running get_logical_error_rate_func with a larger"
                    " max_error_rate."
                )
            probs = _get_error_probs_by_weight(len(self), error_rate, max_error_weight)
            return 1 - probs @ fidelities, np.sqrt(probs**2 @ variances)

        return get_logical_error_rate

    def _estimate_decoding_fidelity_and_variance(
        self, error_weight: int, num_samples: int, decoder: decoders.Decoder
    ) -> tuple[float, float]:
        """Estimate a fidelity and its variance when decoding a fixed number of errors."""
        num_failures = 0
        for _ in range(num_samples):
            # construct an error
            error_locations = random.sample(range(len(self)), error_weight)
            error = np.zeros(len(self), dtype=int)
            error[error_locations] = np.random.choice(range(1, self.field.order), size=error_weight)

            # decode a corrupted all-zero code word
            decoded_word = decoder.decode(error.view(np.ndarray))
            if np.any(decoded_word):
                num_failures += 1

        infidelity = num_failures / num_samples
        variance = infidelity * (1 - infidelity) / num_samples
        return 1 - infidelity, variance


################################################################################
# quantum codes


class QuditCode(AbstractCode):
    """Quantum code for Galois qudits with dimension q = p^m for prime p and integer m.

    A QuditCode is initialized from a parity check matrix H whose rows represent Pauli strings.  If
    all of these Pauli strings commute, then the QuditCode is a stabilizer code, and the rows of H
    are generators of the code's stabilizer group.  Otherwise, the QuditCode is a subsystem code,
    which is equivalent to a stabilizer code in which some logical qudits are promoted to "gauge
    qudits".  In this case, the rows of H are generators of the code's gauge group.

    More specifically, for a QuditCode with block length num_qudits, each row of H is a symplectic
    vector P = [P_x|P_z] of length 2 * num_qudits, where each of P_x and P_z are vectors of length
    num_qudits that indicate the support of X-type and Z-type Pauli operators on the physical qudits
    of the QuditCode.  If P_x[j] = r_x and P_z[j] = r_z, where r_x and r_z are elements of the the
    Galois field GF(q) (for example, GF(2) ~ {0, 1} for qubits), then the Pauli string P addresses
    physical qudit j by the qudit operator X(r_x) Z(r_z), where
    - X(r) = sum_{k=0}^{q-1} |k+r><k| is a shift operator, and
    - Z(r) = sum_{k=0}^{q-1} w^{k r} |k><k| is a phase operator, with w = exp(2 pi i / q).
    Here r and k are not integers, but elements of the Galois field GF(q), which has special rules
    for addition and multiplication when q is not a prime number.

    The matrix H is a "parity check matrix" in the sense that its null space with respect to the
    symplectic inner product ⟨P,Q⟩_s = P_x @ Q_z - P_z @ Q_x = symplectic_conjugate(P) @ Q is the
    space of logical Pauli operators of the QuditCode.

    References:
    - https://errorcorrectionzoo.org/c/galois_into_galois
    - https://errorcorrectionzoo.org/c/galois_stabilizer
    - https://errorcorrectionzoo.org/c/oecc
    Helpful lecture by Gottesman: https://www.youtube.com/watch?v=JWg4zrNAF-g
    """

    _is_subsystem_code: bool
    _dimension: int
    _stabilizer_ops: galois.FieldArray | None = None
    _gauge_ops: galois.FieldArray | None = None
    _logical_ops: galois.FieldArray | None = None

    def __init__(
        self,
        matrix: AbstractCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | None = None,
        *,
        is_subsystem_code: bool | None = None,
    ) -> None:
        """Construct a qudit code from a parity check matrix over a finite field."""
        AbstractCode.__init__(self, matrix, field)
        if is_subsystem_code is not None:
            self._is_subsystem_code = is_subsystem_code
        else:
            self._is_subsystem_code = bool(
                np.any(symplectic_conjugate(self.matrix) @ self.matrix.T)
            )

    def __eq__(self, other: object) -> bool:
        """Equality test between two code instances."""
        return (
            isinstance(other, QuditCode)
            and self.field is other.field
            and np.array_equal(self.matrix, other.matrix)
        )

    def __str__(self) -> str:
        """Human-readable representation of this code."""
        text = f"{self.name} on {len(self)}"
        if self.field.order == 2:
            text += " qubits"
        else:
            text += f" qudits over {self.field_name}"
        text += f", with parity check matrix\n{self.matrix}"
        return text

    @property
    def is_subsystem_code(self) -> bool:
        """Is this code a subsystem code?  If not, it is a stabilizer code."""
        return self._is_subsystem_code

    @functools.cached_property
    def canonicalized(self) -> QuditCode:
        """The same code with its parity matrix in reduced row echelon form."""
        matrix_rref = self.matrix.row_reduce()
        matrix_rref = matrix_rref[np.any(matrix_rref, axis=1), :]
        return QuditCode(matrix_rref, self.field.order, is_subsystem_code=self.is_subsystem_code)

    @staticmethod
    def matrix_to_graph(matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]]) -> nx.DiGraph:
        """Convert a parity check matrix into a Tanner graph."""
        graph = nx.DiGraph()
        matrix = np.reshape(matrix, (len(matrix), 2, -1))
        for row, xz, col in zip(*np.nonzero(matrix)):
            node_check = Node(index=int(row), is_data=False)
            node_qudit = Node(index=int(col), is_data=True)
            graph.add_edge(node_check, node_qudit)

            qudit_op = graph[node_check][node_qudit].get(QuditOperator, QuditOperator())
            vals_xz = list(qudit_op.value)
            vals_xz[xz] += int(matrix[row, xz, col])
            graph[node_check][node_qudit][QuditOperator] = QuditOperator(tuple(vals_xz))

        # remember order of the field, and use Pauli operators if appropriate
        if isinstance(matrix, galois.FieldArray):
            graph.order = type(matrix).order
            if graph.order == 2:
                for _, __, data in graph.edges(data=True):
                    data[Pauli] = Pauli(data[QuditOperator].value)
                    del data[QuditOperator]

        return graph

    @staticmethod
    def graph_to_matrix(graph: nx.DiGraph) -> galois.FieldArray:
        """Convert a Tanner graph into a parity check matrix."""
        num_qudits = sum(1 for node in graph.nodes() if node.is_data)
        num_checks = len(graph.nodes()) - num_qudits
        matrix = np.zeros((num_checks, 2, num_qudits), dtype=int)
        for node_check, node_qudit, data in graph.edges(data=True):
            op = data.get(QuditOperator) or data.get(Pauli)
            matrix[node_check.index, :, node_qudit.index] = op.value
        field = getattr(graph, "order", DEFAULT_FIELD_ORDER)
        return galois.GF(field)(matrix.reshape(num_checks, 2 * num_qudits))

    def get_strings(self) -> list[str]:
        """Parity checks checks of this code, represented by strings."""
        matrix = self.matrix.reshape(self.num_checks, 2, self.num_qudits)
        checks = []
        for check in range(self.num_checks):
            ops = []
            for qudit in range(self.num_qudits):
                val_x = matrix[check, Pauli.X, qudit]
                val_z = matrix[check, Pauli.Z, qudit]
                vals_xz = (val_x, val_z)
                if self.field.order == 2:
                    ops.append(str(Pauli(vals_xz)))
                else:
                    ops.append(str(QuditOperator(vals_xz)))
            checks.append(" ".join(ops))
        return checks

    @staticmethod
    def from_strings(*checks: str, field: int | None = None) -> QuditCode:
        """Construct a QuditCode from the provided parity checks."""
        field = field or DEFAULT_FIELD_ORDER
        check_ops = [check.split() for check in checks]
        num_checks = len(check_ops)
        num_qudits = len(check_ops[0])
        operator: type[Pauli] | type[QuditOperator] = Pauli if field == 2 else QuditOperator

        matrix = np.zeros((num_checks, 2, num_qudits), dtype=int)
        for index, check_op in enumerate(check_ops):
            if len(check_op) != num_qudits:
                raise ValueError(f"Parity checks 0 and {index} have different lengths")
            for qudit, op in enumerate(check_op):
                matrix[index, :, qudit] = operator.from_string(op).value

        return QuditCode(matrix.reshape(num_checks, 2 * num_qudits), field)

    def __len__(self) -> int:
        """The block length of this code."""
        return self.matrix.shape[1] // 2

    @property
    def num_qudits(self) -> int:
        """Number of data qudits in this code."""
        return len(self)

    @property
    def num_qubits(self) -> int:
        """Number of data qubits in this code."""
        if not self.field.order == 2:
            raise ValueError(
                "You asked for the number of qubits in this code, but this code is built out of "
                rf"{self.field.order}-dimensional qudits.\nTry calling {type(self)}.num_qudits."
            )
        return len(self)

    def get_weight(self) -> int:
        """Compute the weight of the largest parity check."""
        matrix_x = self.matrix[:, : len(self)].view(np.ndarray)
        matrix_z = self.matrix[:, len(self) :].view(np.ndarray)
        return np.max(np.count_nonzero(matrix_x | matrix_z, axis=1))

    def get_logical_ops(
        self, pauli: PauliXZ | None = None, *, recompute: bool = False
    ) -> galois.FieldArray:
        """Basis of nontrivial logical Pauli operators for this code.

        Logical operators are represented by a matrix logical_ops with shape (2 * k, 2 * n), where
        k and n are, respectively, the numbers of logical and physical qudits in this code.
        Each row of logical_ops is a vector that represents a logical operator.  The first
        (respectively, second) n entries of this vector indicate the support of _physical_ X-type
        (respectively, Z-type) operators.  Similarly, the first (second) k rows correspond to
        _logical_ X-type (Z-type) operators.  The logical operators at rows j and j+k are dual to
        each other, which is to say that the logical operator at row j commutes with the logical
        operators in all other rows except row j+k.

        If this method is passed a pauli operator (Pauli.X or Pauli.Z), it returns only the logical
        operators of that type.

        Due to the way that logical operators are constructed in this method, logical Z-type
        operators only address physical qudits by physical Z-type operators, while logical X-type
        operators address at least one physical qudits a physical X-type operator, but may
        additionally address physical qudits with physical Z-type operators.

        Logical operators are constructed with the method similar to that in Section 4.1 of
        Gottesman's thesis (arXiv:9705052), generalized for subsystem qudit codes.  The basic
        strategy is to fix the values of the logical operator matrix in the GK sector of the parity
        check matrix when written in standard form (see QuditCode.get_standard_form_data), and then
        fill in the remaining entries of the logical operator matrix as required by parity check
        constraints.
        """
        assert pauli is None or pauli in PAULIS_XZ

        # if requested, retrieve logical operators of one type only
        if pauli is not None:
            return self.get_logical_ops(recompute=recompute).reshape(2, -1, 2 * len(self))[pauli]

        # return logical operators if known and not asked to recompute
        if not (self._logical_ops is None or recompute):
            return self._logical_ops

        # construct the standard-form parity check matrix
        (
            matrix,
            qudit_locs,
            (rows_sx, rows_gx, rows_sz, rows_gz),
            (cols_sx, cols_gx, cols_lx, cols_sz, cols_gz, cols_lz),
        ) = self.get_standard_form_data()
        matrix_x = matrix[:, 0, :]
        matrix_z = matrix[:, 1, :]

        # X/Z support of X/Z logical operators, as column vectors
        logicals_xx = self.field.Zeros((len(self), self.dimension))
        logicals_zz = self.field.Zeros((len(self), self.dimension))

        # "seed" the logical operators in the GK sector
        if not self.is_subsystem_code:
            logicals_xx[cols_lz] = self.field.Identity(self.dimension)
            logicals_zz[cols_lx] = self.field.Identity(self.dimension)

        else:
            cols_gk = sorted(_join_slices(cols_gx, cols_lx))  # indices for all GK columns
            """
            Focusing on the gauge-qudit rows (i.e., constraints) of the parity check matrix, define
                A = matrix_z[rows_gz, cols_gk],
                B = matrix_x[rows_gx, cols_gk],
            and denode the logical operator components in the GK sector by
                U = logicals_xx[cols_gk],
                V = logicals_zz[cols_gk].
            These components need to satisfy the system of matrix equations
                (1) A @ U.T = 0,
                (2) B @ V.T = 0,
                (3) U.T @ V = I.
            Without loss of generality, we can satisfy (1) and (2) by setting
                U = null_space(A).T
                V = null_space(B).T @ M,
            where the matrix M is determined by subsituting U and V back into (3),
                U.T @ W @ M = I.
            """
            mat_U = matrix_z[rows_gz, cols_gk].null_space().T  # type:ignore[attr-defined]
            mat_W = matrix_x[rows_gx, cols_gk].null_space().T  # type:ignore[attr-defined]
            mat_M = np.linalg.inv(mat_U.T @ mat_W)
            logicals_xx[cols_gk] = mat_U
            logicals_zz[cols_gk] = mat_W @ mat_M

        # fill in remaining entries by enforcing parity check constraints
        logicals_xx[cols_sz] = -matrix_z[rows_sz] @ logicals_xx
        logicals_zz[cols_sx] = -matrix_x[rows_sx] @ logicals_zz

        # Z support of X-type logicals, as column vectors
        logicals_xz = self.field.Zeros((len(self), self.dimension))
        logicals_xz[cols_lx] = self.field.Identity(self.dimension)
        logicals_xz[cols_gx] = -matrix_x[rows_gx] @ logicals_xz + matrix_z[rows_gx] @ logicals_xx
        logicals_xz[cols_sx] = -matrix_x[rows_sx] @ logicals_xz + matrix_z[rows_sx] @ logicals_xx

        # full X and Z logicals as row vectors
        logicals_x = np.vstack([logicals_xx, logicals_xz]).T
        logicals_z = np.vstack([np.zeros_like(logicals_zz), logicals_zz]).T

        # move qudits back to their original locations, save logicals, and return
        permutation = np.argsort(qudit_locs)
        logicals_x = logicals_x.reshape(-1, len(self))[:, permutation].reshape(-1, 2 * len(self))
        logicals_z = logicals_z.reshape(-1, len(self))[:, permutation].reshape(-1, 2 * len(self))
        self._logical_ops = np.vstack([logicals_x, logicals_z])  # type:ignore[assignment]
        return self._logical_ops  # type:ignore[return-value]

    def get_standard_form_data(
        self,
    ) -> tuple[
        npt.NDArray[np.int_],  # standard-form matrix, with shape (-1, 2, len(self))
        npt.NDArray[np.int_],  # qudit locations
        tuple[slice, slice, slice, slice],  # row sectors
        tuple[slice, Slice, Slice, slice, Slice, Slice],  # column sectors
    ]:
        """Construct the standard form of the parity check matrix with Gaussian elimination.

        The standard form of the parity check is the block matrix

        ⌈  I   ·   ·   ·  |  ·   ·   ·   ·  ⌉ S_X --> rows_sx (X-type stabilizers)
        |      ·   I   ·  |  ·   ·   ·   ·  | G_X --> rows_gx (X-type gauge ops)
        |                 |  ·   I   ·   ·  | S_Z --> rows_sz (Z-type stabilizers)
        ⌊                 |  ·       I   ·  ⌋ G_Z --> rows_gz (Z-type gauge ops)
          S_X S_Z G_X L_X   S_X S_Z G_Z L_Z
           |   |   |   |     |   |   |   |
           |   |   |   |     |   |   |   └----------> cols_lz (# of columns = # of logical qudits)
           |   |   |   |     |   |   └--------------> cols_gz (Z-type gauge pivots)
           |   |   |   |     |   └------------------> cols_sz (Z-type stabilizer pivots)
           |   |   |   |     └----------------------> cols_sx (X-type stabilizer pivots)
           |   |   |   └----------------------------> cols_lx (# of columns = # of logical qudits)
           |   |   └--------------------------------> cols_gx (X-type gauge pivots)
           |   └------------------------------------> cols_sz (Z-type stabilizer pivots)
           └----------------------------------------> cols_sx (X-type stabilizer pivots)

        Here I is an identity matrix of an appropriate size, and dots (·) indicate nonzero blocks
        of the matrix.  Each row sector is associated with sets of linearly independent stabilizers
        or gauge operators, though the gauge operators are generally not necessarily sorted into
        conjugate pairs (as in self.get_gauge_ops() and self.get_logical_ops()).

        For convenience, the standard-form matrix is returned with shape (-1, 2, len(self)), such
        that, for example, matrix[r, 1, :] indicates the Pauli-Z support of parity check r.

        In addition to the standard-form matrix, this method returns a 1-D array qudit_locs, for
        which qudit_locs[j] is the location of qudit j in the standard-form matrix.  The original
        parity check matrix (modulo elementary row operations and array reshaping) is
        matrix[:, :, np.argsort(qudit_locs)].  As a sanity check, the following test should pass:
            matrix_2d = matrix[:, :, np.argsort(qudit_locs)].reshape(-1, 2 * len(self))
            assert np.array_equal(matrix_2d.row_reduce(), self.canonicalized.matrix)

        Finally, this method also returns slices (index sets) for all row and column sectors, which
        enables selecting blocks of the parity check matrix with, say, matrix[rows_sx, 1, cols_lz].

        One subtlety to note is that columns of the standard-form matrix may not be in the same
        order as that suggested by the visualization above, but blocks retrieved by the index sets
        are guaranteed to be consistent with the placement of identity matrices, which is to say
        that each of
            matrix[rows_sx, 0, cols_sx]
            matrix[rows_gx, 0, cols_gx]
            matrix[rows_sz, 1, cols_sz]
            matrix[rows_gz, 1, cols_gz]
        is guaranteed to be an identity matrix.
        """
        matrix: npt.NDArray[np.int_]
        cols_lx: Slice
        cols_lz: Slice

        def _with_permuted_qudits(
            matrix: npt.NDArray[np.int_], permutation: Slice
        ) -> npt.NDArray[np.int_]:
            """Permute the qudits of a parity check matrix."""
            return matrix.reshape(-1, len(self))[:, permutation].reshape(matrix.shape)

        if not self.is_subsystem_code:
            # keep track of qudit locations as we swap them around
            qudit_locs = np.arange(len(self), dtype=int)

            # row reduce and identify pivots in the X sector
            matrix = self.canonicalized.matrix
            all_pivots_x = _first_nonzero_cols(matrix)
            pivots_x = all_pivots_x[all_pivots_x < len(self)]

            # move the X pivots to the back
            other_x = [qq for qq in range(len(self)) if qq not in pivots_x]
            permutation = other_x + list(pivots_x)
            matrix = _with_permuted_qudits(matrix, permutation)
            qudit_locs = qudit_locs[permutation]

            # row reduce and identify pivots in the Z sector
            sub_matrix = matrix[len(pivots_x) :, len(self) :]
            sub_matrix = ClassicalCode(sub_matrix).canonicalized.matrix
            matrix[len(pivots_x) :, len(self) :] = sub_matrix
            all_pivots_z = _first_nonzero_cols(sub_matrix)
            pivots_z = all_pivots_z[: len(self) - len(pivots_x) - self.dimension]

            # move the stabilizer Z pivots to the back
            other_z = [qq for qq in range(len(self)) if qq not in pivots_z]
            permutation = other_z + list(pivots_z)
            matrix = _with_permuted_qudits(matrix, permutation)
            qudit_locs = qudit_locs[permutation]

            # some helpful numbers
            num_stabs_x = len(pivots_x)
            num_stabs_z = len(pivots_z)
            num_logicals = len(self) - num_stabs_x - num_stabs_z

            # row/column sectors of the parity check matrix
            rows_sx = slice(num_stabs_x)
            rows_sz = slice(rows_sx.stop, rows_sx.stop + num_stabs_z)
            cols_lx = cols_lz = slice(num_logicals)
            cols_sx = slice(cols_lx.stop, cols_lx.stop + num_stabs_x)
            cols_sz = slice(cols_sx.stop, cols_sx.stop + num_stabs_z)

            # fill in empty gauge sectors
            rows_gx = rows_gz = cols_gx = cols_gz = slice(0)

        else:
            # X-type and Z-type stabilizers in standard form
            stabilizer_ops = self.get_stabilizer_ops()
            code = QuditCode(stabilizer_ops, is_subsystem_code=False)
            (
                standard_stabilizer_ops,
                qudit_locs,
                (rows_sx, _, rows_sz, _),
                (cols_sx, _, _, cols_sz, _, cols_gk),
            ) = code.get_standard_form_data()
            stabilizers_x = standard_stabilizer_ops[rows_sx].reshape(-1, 2 * len(self))
            stabilizers_z = standard_stabilizer_ops[rows_sz, 1, :]
            cols_gk = _join_slices(cols_gk)  # convert into indexable array

            # some helpful numbers
            num_stabs_x = len(stabilizers_x)
            num_stabs_z = len(stabilizers_z)
            num_gauges = self.gauge_dimension
            num_logicals = len(self) - num_stabs_x - num_stabs_z - num_gauges

            # canonicalized parity check matrices with qudits in the same order as above
            checks = _with_permuted_qudits(self.canonicalized.matrix, qudit_locs)
            checks_x = checks[: num_stabs_x + num_gauges]
            checks_z = checks[num_stabs_x + num_gauges :, len(self) :]

            # row reduce X-type stabilizers + parity checks to ensure that gauge ops at the bottom
            permutation_x = _join_slices(cols_sx, cols_gk, cols_sz)
            matrix_x = _with_permuted_qudits(np.vstack([stabilizers_x, checks_x]), permutation_x)
            matrix_x = ClassicalCode(matrix_x).canonicalized.matrix
            pivots_gx = _first_nonzero_cols(matrix_x)[num_stabs_x:] - num_stabs_x
            matrix_x = _with_permuted_qudits(matrix_x, np.argsort(permutation_x))

            # row reduce Z-type stabilizers + parity checks to ensure that gauge ops at the bottom
            permutation_z = _join_slices(cols_sz, cols_gk, cols_sx)
            matrix_z = _with_permuted_qudits(np.vstack([stabilizers_z, checks_z]), permutation_z)
            matrix_z = ClassicalCode(matrix_z).canonicalized.matrix
            pivots_gz = _first_nonzero_cols(matrix_z)[num_stabs_z:] - num_stabs_z
            matrix_z = _with_permuted_qudits(matrix_z, np.argsort(permutation_z))

            """
            Row reducing the combiner stabilizer + gauge matrices added gauge ops to stabilizers to
            zero out entries above the gauge-pivot columns.  Remove the added gauge operators.
            """
            matrix_x[:num_stabs_x] += (
                stabilizers_x[:num_stabs_x, pivots_gx] @ matrix_x[num_stabs_x:]
            )
            matrix_z[:num_stabs_z] += (
                stabilizers_z[:num_stabs_z, pivots_gz] @ matrix_z[num_stabs_z:]
            )

            # full parity check matrix in standard form
            matrix = np.vstack([matrix_x, np.hstack([np.zeros_like(matrix_z), matrix_z])])

            # identify row sectors for the standard-form matrix
            rows_sx = slice(num_stabs_x)
            rows_gx = slice(rows_sx.stop, rows_sx.stop + num_gauges)
            rows_sz = slice(rows_gx.stop, rows_gx.stop + num_stabs_z)
            rows_gz = slice(rows_sz.stop, rows_sz.stop + num_gauges)

            # split logical vs. gauge column sectors
            cols_gx = cols_gk[pivots_gx]
            cols_gz = cols_gk[pivots_gz]
            cols_lx = [qq for qq in cols_gk if qq not in cols_gx]  # type:ignore[operator]
            cols_lz = [qq for qq in cols_gk if qq not in cols_gz]  # type:ignore[operator]

        matrix = matrix.reshape(-1, 2, len(self))
        return (
            matrix,
            qudit_locs,
            (rows_sx, rows_gx, rows_sz, rows_gz),
            (cols_sx, cols_gx, cols_lx, cols_sz, cols_gz, cols_lz),
        )

    def set_logical_ops(
        self, logical_ops: npt.NDArray[np.int_] | Sequence[Sequence[int]], *, validate: bool = True
    ) -> None:
        """Set the logical operators of this code to the provided logical operators."""
        logical_ops = self.field(logical_ops)
        if validate:
            dimension = len(logical_ops) // 2
            logical_ops_x = logical_ops[:dimension]
            logical_ops_z = logical_ops[dimension:]
            inner_products = symplectic_conjugate(logical_ops_x) @ logical_ops_z.T
            if not np.array_equal(inner_products, np.eye(dimension, dtype=int)):
                raise ValueError("The given logical operators have incorrect commutation relations")
            if np.any(symplectic_conjugate(self.matrix) @ logical_ops.T):
                raise ValueError("The given logical operators violate parity checks")
            if dimension != self.dimension:
                raise ValueError("An incorrect number of logical operators was provided")
        self._logical_ops = logical_ops

    def get_stabilizer_ops(
        self, pauli: PauliXZ | None = None, *, canonicalized: bool = False
    ) -> galois.FieldArray:
        """Basis of stabilizer group generators for this code.

        If canonicalized is True, guarantee that the stabilizer matrix is canonicalized (i.e., row
        reduced) such that its rows are a minimal generating set for the stabilizer group.
        """
        assert pauli is None or pauli in PAULIS_XZ

        # if requested, retrieve stabilizer operators of one type only
        if pauli is not None:
            stabilizer_ops = self.get_stabilizer_ops()
            pivots_x = _first_nonzero_cols(stabilizer_ops) < len(self)
            return stabilizer_ops[pivots_x if pauli is Pauli.X else ~pivots_x]

        # return stabilizers if known
        if not self.is_subsystem_code:
            return self.matrix if not canonicalized else self.canonicalized.matrix
        if self._stabilizer_ops is not None:
            return self._stabilizer_ops

        stabs_and_gauges = self.canonicalized.matrix
        stabs_and_logs = symplectic_conjugate(stabs_and_gauges).null_space()
        stabs_and_gauges_and_logs = np.vstack([stabs_and_gauges, stabs_and_logs])
        assert isinstance(stabs_and_gauges_and_logs, galois.FieldArray)

        self._stabilizer_ops = symplectic_conjugate(stabs_and_gauges_and_logs).null_space()
        return self._stabilizer_ops

    def get_gauge_ops(self, pauli: PauliXZ | None = None) -> galois.FieldArray:
        """Basis of nontrivial logical Pauli operators for the gauge qudits of this code.

        Nontrivial logical Pauli operators for the gauge qudits are organized similarly to the
        logical Pauli operators computed by QuditCode.get_logical_ops.
        """
        assert pauli is None or pauli in PAULIS_XZ

        if not self.is_subsystem_code:
            return self.field.Zeros((0, 2 * len(self)))

        # if requested, retrieve gauge operators of one type only
        if pauli is not None:
            return self.get_gauge_ops().reshape(2, -1, 2 * len(self))[pauli]

        # return gauge operators if known
        if self._gauge_ops is not None:
            return self._gauge_ops

        self._gauge_ops = self.get_dual_subsystem_code().get_logical_ops()
        return self._gauge_ops

    def get_dual_subsystem_code(self) -> QuditCode:
        """Get the subsystem code that swaps gauge and logical qudits of this code."""
        matrix = np.vstack([self.get_stabilizer_ops(), self.get_logical_ops()])
        return QuditCode(matrix, is_subsystem_code=self.dimension != 0)

    @functools.cached_property
    def dimension(self) -> int:
        """The number of logical qudits encoded by this code."""
        if not self.is_subsystem_code:
            return len(self) - self.rank
        num_stabs = len(self.get_stabilizer_ops(canonicalized=True))
        return len(self) - (self.rank + num_stabs) // 2

    @functools.cached_property
    def gauge_dimension(self) -> int:
        """The number of gauge qudits in this code."""
        if not self.is_subsystem_code:
            return 0
        num_stabs = len(self.get_stabilizer_ops(canonicalized=True))
        return (self.rank - num_stabs) // 2

    def get_code_params(
        self, *, bound: int | bool | None = None, **decoder_args: Any
    ) -> tuple[int, int, int | float]:
        """Compute the parameters of this code: [n,k,d].

        Here:
        - n is the number of data qudits
        - k is the number of encoded ("logical") qudits
        - d is the code distance

        If `bound is None`, compute an exact code distance by brute force.  Otherwise, compute an
        upper bound using `bound` trials of a a randomized algorithm.  For a detailed explanation,
        see the `get_one_distance_bound` method.

        Keyword arguments are passed to the calculation of code distance.
        """
        distance = self.get_distance(bound=bound, **decoder_args)
        return len(self), self.dimension, distance

    def get_distance(self, *, bound: int | bool | None = None, **decoder_args: Any) -> int | float:
        """Compute (or upper bound) the minimum weight of nontrivial logical operators.

        If `bound is None`, compute an exact code distance by brute force.  Otherwise, compute
        upper bounds using a randomized algorithm and minimize over `bound` trials.  For a detailed
        explanation, see `get_one_distance_bound`.

        Additional arguments, if applicable, are passed to a decoder in `get_one_distance_bound`.
        """
        if not bound:
            return self.get_distance_exact()
        return self.get_distance_bound(num_trials=int(bound), **decoder_args)

    def get_distance_exact(self) -> int | float:
        """Compute the minimum weight of nontrivial logical operators by brute force."""
        if (known_distance := self.get_distance_if_known()) is not None:
            return known_distance

        logical_ops = self.get_logical_ops()
        stabilizer_ops = self.get_stabilizer_ops(canonicalized=True)
        if self.field.order == 2:
            distance = get_distance_quantum(
                logical_ops.view(np.ndarray).astype(np.uint8),
                stabilizer_ops.view(np.ndarray).astype(np.uint8),
            )
        else:
            warnings.warn(
                "Computing the exact distance of a non-binary code may take a (very) long time"
            )
            distance = len(self)
            code_logical_ops = ClassicalCode.from_generator(logical_ops)
            code_stabilizer_ops = ClassicalCode.from_generator(stabilizer_ops)
            for word_l, word_s in itertools.product(
                code_logical_ops.iter_words(skip_zero=True),
                code_stabilizer_ops.iter_words(),
            ):
                word = word_l + word_s
                support_x = word[: len(self)].view(np.ndarray)
                support_z = word[len(self) :].view(np.ndarray)
                distance = min(distance, np.count_nonzero(support_x | support_z))

        self._exact_distance = int(distance)
        return distance

    def get_distance_if_known(self) -> int | float | None:
        """Retrieve exact distance, if known.  Otherwise return None."""
        # the distance of dimension-0 codes is undefined
        if self.dimension == 0:
            self._exact_distance = np.nan

        return self._exact_distance

    def get_distance_bound(
        self, num_trials: int = 1, *, cutoff: int | None = None, **decoder_args: Any
    ) -> int | float:
        """Compute an upper bound on code distance by minimizing many individual upper bounds.

        If passed a cutoff, don't bother trying to find distances less than the cutoff.

        Additional arguments, if applicable, are passed to a decoder in `get_one_distance_bound`.
        """
        if (known_distance := self.get_distance_if_known()) is not None:
            return known_distance

        min_bound = len(self)
        for _ in range(num_trials):
            if cutoff and min_bound <= cutoff:
                break
            new_bound = self.get_one_distance_bound(**decoder_args)
            min_bound = int(min(min_bound, new_bound))
        return min_bound

    def get_one_distance_bound(self, **decoder_args: Any) -> int:
        """Use a randomized algorithm to compute a single upper bound on code distance."""
        raise NotImplementedError(
            "Monte Carlo distance bound calculation is not implemented for a general QuditCode"
        )

    def conjugated(self, qudits: slice | Sequence[int] | None = None) -> QuditCode:
        """Apply local Fourier transforms to data qudits, swapping X-type and Z-type operators."""
        if qudits is None:
            qudits = getattr(self, "_default_conjugate", ())
        matrix = self.matrix.copy().reshape(-1, 2, len(self))
        matrix[:, :, qudits] = matrix[:, ::-1, qudits]
        matrix = matrix.reshape(-1, 2 * len(self))
        code = QuditCode(matrix, field=self.field.order, is_subsystem_code=self.is_subsystem_code)

        if self._logical_ops is not None:
            logical_ops = self._logical_ops.copy().reshape(-1, 2, len(self))
            logical_ops[:, :, qudits] = logical_ops[:, ::-1, qudits]
            logical_ops = logical_ops.reshape(-1, 2 * len(self))
            code.set_logical_ops(logical_ops)

        return code

    def deformed(
        self, circuit: str | stim.Circuit, *, preserve_logicals: bool = False
    ) -> QuditCode:
        """Deform a code by the given circuit.

        If preserve_logicals is True, preserve the logical operators of the original code (or throw
        an error if the original logical operators are invalid for the deformed code).  Otherwise,
        transform the logical operators as well.
        """
        if not self.field.order == 2:
            raise ValueError("Code deformation is only supported for qubit codes")

        # convert the physical circuit into a tableau
        identity = stim.Circuit(f"I {len(self) - 1}")
        circuit = stim.Circuit(circuit) if isinstance(circuit, str) else circuit
        tableau = (circuit + identity).to_tableau()

        def deform_strings(strings: IntegerArray) -> IntegerArray:
            """Deform the given Pauli strings."""
            new_strings = []
            for check in strings:
                string = op_to_string(check)
                xs_zs = tableau(string).to_numpy()
                new_strings.append(np.concatenate(xs_zs))
            return self.field(new_strings)

        # deform this code by transforming its stabilizers
        matrix = deform_strings(self.matrix)
        new_code = QuditCode(matrix, is_subsystem_code=self.is_subsystem_code)

        # preserve or update logical operators, as applicable
        if preserve_logicals:
            new_code.set_logical_ops(self.get_logical_ops())
        elif self._logical_ops is not None:
            new_code.set_logical_ops(deform_strings(self.get_logical_ops()))

        return new_code

    @staticmethod
    def stack(*codes: QuditCode, inherit_logicals: bool = True) -> QuditCode:
        """Stack the given qudit codes.

        The stacked code is obtained by having the input codes act on disjoint sets of bits.
        Stacking two codes with parameters [n_1, k_1, d_1] and [n_2, k_2, d_2], for example, results
        in a single code with parameters [n_1 + n_2, k_1 + k_2, min(d_1, d_2)].
        """
        codes_x = [ClassicalCode(code.matrix.reshape(-1, 2, len(code))[:, 0, :]) for code in codes]
        codes_z = [ClassicalCode(code.matrix.reshape(-1, 2, len(code))[:, 1, :]) for code in codes]
        code_x = ClassicalCode.stack(*codes_x)
        code_z = ClassicalCode.stack(*codes_z)
        matrix = np.hstack([code_x.matrix, code_z.matrix])
        is_subsystem_code = any(code.is_subsystem_code for code in codes)
        code = QuditCode(matrix, is_subsystem_code=is_subsystem_code)
        if inherit_logicals:
            logicals_xx = [
                QuditCode.get_logical_ops(code, Pauli.X)[:, : len(code)] for code in codes
            ]
            logicals_zx = [
                QuditCode.get_logical_ops(code, Pauli.Z)[:, : len(code)] for code in codes
            ]
            logicals_xz = [
                QuditCode.get_logical_ops(code, Pauli.X)[:, len(code) :] for code in codes
            ]
            logicals_zz = [
                QuditCode.get_logical_ops(code, Pauli.Z)[:, len(code) :] for code in codes
            ]
            logical_ops = np.block(
                [
                    [scipy.linalg.block_diag(*logicals_xx), scipy.linalg.block_diag(*logicals_xz)],
                    [scipy.linalg.block_diag(*logicals_zx), scipy.linalg.block_diag(*logicals_zz)],
                ]
            )
            code.set_logical_ops(logical_ops)
        return code

    @staticmethod
    def concatenate(
        inner: QuditCode,
        outer: QuditCode,
        outer_physical_to_inner_logical: Mapping[int, int] | Sequence[int] | None = None,
        *,
        inherit_logicals: bool = True,
    ) -> QuditCode:
        """Concatenate two qudit codes.

        The concatenated code uses the logical qudits of the "inner" code as the physical qudits of
        the "outer" code, with outer_physical_to_inner_logical defining the map from outer physical
        qudit index to inner logical qudit index.

        This method nominally assumes that len(outer_physical_to_inner_logical) is equal to both the
        number of logical qudits of the inner code and the number of physical qudits of the outer
        code.  If len(outer_physical_to_inner_logical) is larger than the number of inner logicals
        or outer physicals, then copies of the inner and outer codes are used (stacked together) to
        match the expected number of "intermediate" qudits.  If no outer_physical_to_inner_logical
        mapping is provided, then this method "interleaves" intermediate qubits, using each logical
        qubit of an inner block as a physical qubit of a different outer code block.

        If inherit_logicals is True, use the logical operators of the outer code as the logical
        operators of the concatenated code.  Otherwise, logical operators of the concatenated code
        get recomputed from scratch.
        """
        # stack copies of the inner and outer codes (if necessary) and permute inner logicals
        inner, outer = QuditCode._standardize_concatenation_inputs(
            inner, outer, outer_physical_to_inner_logical
        )
        is_subsystem_code = inner.is_subsystem_code or outer.is_subsystem_code

        """
        Parity checks inherited from the outer code are nominally defined in terms of their support
        on logical operators of the inner code.  Expand these parity checks into their support on
        the physical qudits of the inner code.
        """
        outer_checks = outer.matrix @ inner.get_logical_ops()

        # combine parity checks of the inner and outer codes
        code = QuditCode(
            np.vstack([inner.matrix, outer_checks]), is_subsystem_code=is_subsystem_code
        )

        if inherit_logicals:
            code._logical_ops = outer.get_logical_ops() @ inner.get_logical_ops()
        return code

    @staticmethod
    def _standardize_concatenation_inputs(
        inner: QuditCode,
        outer: QuditCode,
        outer_physical_to_inner_logical: Mapping[int, int] | Sequence[int] | None,
    ) -> tuple[QuditCode, QuditCode]:
        """Helper function for code concatenation.

        This method...
        - stacks copies of the inner and outer codes as necessary to make the number of logical
          qudits of the inner code equal to the number of physical qudits of the outer code, and
        - permutes logical qudits of the inner code according to outer_physical_to_inner_logical.
          If no outer_physical_to_inner_logical mapping is provided, then this method "interleaves"
          intermediate qubits, using each logical qubit of an inner block as a physical qubit of a
          different outer code block.
        """
        if inner.field is not outer.field:
            raise ValueError("Cannot concatenate codes over different fields")

        # convert outer_physical_to_inner_logical into a tuple that we can use to permute an array
        if outer_physical_to_inner_logical is None:
            outer_physical_to_inner_logical = tuple(
                np.arange(len(outer) * inner.dimension, dtype=int)
                .reshape(len(outer), inner.dimension)
                .T.ravel()
            )
        else:
            num_qudits = len(outer_physical_to_inner_logical)
            if num_qudits % inner.dimension or num_qudits % len(outer):
                raise ValueError(
                    "Code concatenation requires the number of qudits mapped by"
                    f" outer_physical_to_inner_logical ({num_qudits}) to be divisible by the number"
                    f" of logical qudits of the inner code ({inner.dimension}) and the number of"
                    f" physical qudits of the outer code ({len(outer)})"
                )
            outer_physical_to_inner_logical = tuple(
                outer_physical_to_inner_logical[qq]
                for qq in range(len(outer_physical_to_inner_logical))
            )

        # stack copies of the inner and outer codes, if necessary
        if (num_inner_blocks := len(outer_physical_to_inner_logical) // inner.dimension) > 1:
            inner = inner.stack(*[inner] * num_inner_blocks)
        if (num_outer_blocks := len(outer_physical_to_inner_logical) // len(outer)) > 1:
            outer = outer.stack(*[outer] * num_outer_blocks)

        # permute logical operators of the inner code
        inner._logical_ops = (
            inner.get_logical_ops()  # type:ignore[assignment]
            .reshape(2, inner.dimension, -1)[:, outer_physical_to_inner_logical, :]
            .reshape(2 * inner.dimension, -1)
        )

        return inner, outer


class CSSCode(QuditCode):
    """QuditCode with separate X-type and Z-type parity checks.

    A CSSCode is defined from two classical codes with parity check matrices H_x and H_z, whose rows
    indicate, respectively, the support of X-type Pauli strings that witness Z-type errors, and
    Z-type Pauli strings that witness X-type errors.  The full parity check matrix of a CSSCode is
    ⌈ H_x,  0  ⌉
    ⌊  0 , H_z ⌋.

    If all parity checks of a CSSCode commute, H_x @ H_z.T == 0, then the CSSCode is a stabilizer
    code; otherwise, the CSSCode is a subsystem code.

    References:
    - https://errorcorrectionzoo.org/c/galois_subsystem_css
    - https://errorcorrectionzoo.org/c/galois_css
    """

    _code_x: ClassicalCode
    _code_z: ClassicalCode
    _exact_distance_x: int | float | None = None
    _exact_distance_z: int | float | None = None
    _equal_distance_xz: bool

    def __init__(
        self,
        code_x: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_z: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | None = None,
        *,
        is_subsystem_code: bool | None = None,
        promise_equal_distance_xz: bool = False,  # do X and Z logicals have equal minimum weight?
    ) -> None:
        """Build a CSSCode from classical subcodes that specify X-type and Z-type parity checks."""
        self._code_x = ClassicalCode(code_x, field)  # X-type parity checks, measuring Z-type errors
        self._code_z = ClassicalCode(code_z, field)  # Z-type parity checks, measuring X-type errors
        self._field = self.code_x.field
        if len(self.code_x) != len(self.code_z) or self.code_x.field is not self.code_z.field:
            raise ValueError("The sub-codes provided for this CSSCode are incompatible")

        if is_subsystem_code is not None:
            self._is_subsystem_code = is_subsystem_code
        else:
            self._is_subsystem_code = bool(np.any(self.matrix_x @ self.matrix_z.T))
        self._equal_distance_xz = promise_equal_distance_xz or self.code_x == self.code_z

        if self._exact_distance_x is not None and self._exact_distance_z is not None:
            self._exact_distance = min(self._exact_distance_x, self._exact_distance_z)

    def __eq__(self, other: object) -> bool:
        """Equality test between two code instances."""
        return (
            isinstance(other, type(self))
            and self.field is other.field
            and np.array_equal(self.code_x.matrix, other.code_x.matrix)
            and np.array_equal(self.code_z.matrix, other.code_z.matrix)
        )

    def __str__(self) -> str:
        """Human-readable representation of this code."""
        text = ""
        if self.field.order == 2:
            text += f"{self.name} on {len(self)} qubits"
        else:
            text += f"{self.name} on {len(self)} qudits over {self.field_name}"
        text += f"\nX-type parity checks:\n{self.matrix_x}"
        text += f"\nZ-type parity checks:\n{self.matrix_z}"
        return text

    @property
    def code_x(self) -> ClassicalCode:
        """The classical code of X-type parity checks."""
        return self._code_x

    @property
    def code_z(self) -> ClassicalCode:
        """The classical code of Z-type parity checks."""
        return self._code_z

    @property
    def matrix_x(self) -> galois.FieldArray:
        """X-type parity checks."""
        return self.code_x.matrix

    @property
    def matrix_z(self) -> galois.FieldArray:
        """Z-type parity checks."""
        return self.code_z.matrix

    def get_code(self, pauli: PauliXZ) -> ClassicalCode:
        """Retrieve the classical code of stabilizers of a given type."""
        assert pauli in PAULIS_XZ
        return self.code_x if pauli is Pauli.X else self.code_z

    def get_matrix(self, pauli: PauliXZ) -> galois.FieldArray:
        """Retrieve the classical code of stabilizers of a given type."""
        assert pauli in PAULIS_XZ
        return self.matrix_x if pauli is Pauli.X else self.matrix_z

    @functools.cached_property
    def matrix(self) -> galois.FieldArray:
        """Overall parity check matrix."""
        matrix = np.block(
            [
                [self.matrix_x, np.zeros_like(self.matrix_x)],
                [np.zeros_like(self.matrix_z), self.matrix_z],
            ]
        )
        return self.field(matrix)

    @functools.cached_property
    def canonicalized(self) -> CSSCode:
        """The same code with its parity matrices in reduced row echelon form."""
        return CSSCode(
            self.code_x.canonicalized,
            self.code_z.canonicalized,
            is_subsystem_code=self.is_subsystem_code,
        )

    @staticmethod
    def equiv(code_a: AbstractCode, code_b: AbstractCode) -> bool:
        """Do the two codes have the same parity checks?"""
        if isinstance(code_a, CSSCode) and isinstance(code_b, CSSCode):
            return ClassicalCode.equiv(code_a.code_x, code_b.code_x) and ClassicalCode.equiv(
                code_a.code_z, code_b.code_z
            )
        return AbstractCode.equiv(code_a, code_b)

    def __len__(self) -> int:
        """Number of data qudits in this code."""
        return self.matrix_x.shape[1]

    @property
    def num_checks_x(self) -> int:
        """Number of X-type parity checks in this code."""
        return self.matrix_x.shape[0]

    @property
    def num_checks_z(self) -> int:
        """Number of Z-type parity checks in this code."""
        return self.matrix_z.shape[0]

    @property
    def num_checks(self) -> int:
        """Number of parity checks in this code."""
        return self.num_checks_x + self.num_checks_z

    @functools.cached_property
    def rank(self) -> int:
        """Rank of this code's parity check matrix.

        Equivalently, the number of linearly independent parity checks in this code.
        """
        return self.code_x.rank + self.code_z.rank

    def get_logical_ops(
        self, pauli: PauliXZ | None = None, *, symplectic: bool = False, recompute: bool = False
    ) -> galois.FieldArray:
        """Basis of nontrivial logical Pauli operators for this code.

        Logical operators are represented by a matrix logical_ops with shape (2 * k, 2 * n), where
        k and n are, respectively, the numbers of logical and physical qudits in this code.
        Each row of logical_ops is a vector that represents a logical operator.  The first
        (respectively, second) n entries of this vector indicate the support of _physical_ X-type
        (respectively, Z-type) operators.  Similarly, the first (second) k rows correspond to
        _logical_ X-type (Z-type) operators.  The logical operators at rows j and j+k are dual to
        each other, which is to say that the logical operator at row j commutes with the logical
        operators in all other rows except row j+k.

        If this method is passed a pauli operator (Pauli.X or Pauli.Z), it returns only the logical
        operators of that type.  This matrix has shape (k, n) by default, but is expanded into a
        matrix with shape (k, 2 * n) if this method is called with symplectic=True.

        Logical X-type operators only address physical qudits by physical X-type operators, and
        logical Z-type operators only address physical qudits by physical Z-type operators.

        Logical operators are constructed with the method similar to that in Section 4.1 of
        Gottesman's thesis (arXiv:9705052), generalized for subsystem qudit codes.  The basic
        strategy is to fix the values of the logical operator matrix in the GK sector of the parity
        check matrix when written in standard form (see QuditCode.get_standard_form_data), and then
        fill in the remaining entries of the logical operator matrix as required by parity check
        constraints.
        """
        assert pauli is None or pauli in PAULIS_XZ

        # if requested, retrieve logical operators of one type only
        if pauli is not None:
            shape: tuple[int, ...]
            index: list[PauliXZ | slice]
            if symplectic:
                shape = (2, self.dimension, 2 * len(self))
                index = [pauli, slice(None), slice(None)]
            else:
                shape = (2, self.dimension, 2, len(self))
                index = [pauli, slice(None), pauli, slice(None)]
            return self.get_logical_ops(recompute=recompute).reshape(shape)[tuple(index)]  # type:ignore[return-value]

        # return logical operators if known and not asked to recompute
        if not (self._logical_ops is None or recompute):
            return self._logical_ops

        # construct the standard-form parity check matrices
        (
            matrix_x,
            matrix_z,
            qudit_locs,
            (rows_sx, rows_gx, rows_sz, rows_gz),
            (cols_sx, cols_gx, cols_lx, cols_sz, cols_gz, cols_lz),
        ) = self.get_standard_form_data_xz()

        # X/Z support of X/Z logical operators, as column vectors
        logicals_x = self.field.Zeros((len(self), self.dimension))
        logicals_z = self.field.Zeros((len(self), self.dimension))

        # "seed" the logical operators in the GK sector
        if not self.is_subsystem_code:
            logicals_x[cols_lz] = self.field.Identity(self.dimension)
            logicals_z[cols_lx] = self.field.Identity(self.dimension)

        else:
            # see QuditCode.get_logical_ops for an explanation of what's happening here
            cols_gk = sorted(_join_slices(cols_gx, cols_lx))
            mat_U = matrix_z[rows_gz, cols_gk].null_space().T  # type:ignore[attr-defined]
            mat_W = matrix_x[rows_gx, cols_gk].null_space().T  # type:ignore[attr-defined]
            mat_M = np.linalg.inv(mat_U.T @ mat_W)
            logicals_x[cols_gk] = mat_U
            logicals_z[cols_gk] = mat_W @ mat_M

        # fill in remaining entries by enforcing parity check constraints
        logicals_x[cols_sz] = -matrix_z[rows_sz] @ logicals_x
        logicals_z[cols_sx] = -matrix_x[rows_sx] @ logicals_z

        # move qudits back to their original locations, save logicals, and return
        permutation = np.argsort(qudit_locs)
        logicals_x = logicals_x[permutation]
        logicals_z = logicals_z[permutation]
        self._logical_ops = self.field(scipy.linalg.block_diag(logicals_x.T, logicals_z.T))
        return self._logical_ops

    def get_standard_form_data_xz(
        self,
    ) -> tuple[
        npt.NDArray[np.int_],  # standard-form matrix_x, with shape (self.dimension, len(self))
        npt.NDArray[np.int_],  # standard-form matrix_z, with shape (self.dimension, len(self))
        npt.NDArray[np.int_],  # qudit locations
        tuple[slice, slice, slice, slice],  # row sectors
        tuple[slice, Slice, Slice, slice, Slice, Slice],  # column sectors
    ]:
        """Construct the standard form X/Z parity check matrices with Gaussian elimination.

        See QuditCode.get_standard_form_data for additional information.  The primary difference
        here is that this method returns the standard forms of matrix_x and matrix_z separately.
        """
        cols_lx: Slice
        cols_lz: Slice

        if not self.is_subsystem_code:
            # keep track of qudit locations as we swap them around
            qudit_locs = np.arange(len(self), dtype=int)

            # initialize matrix_x and matrix_z
            matrix_x = self.canonicalized.matrix_x
            matrix_z = self.canonicalized.matrix_z

            # identify pivots in the X sector, and move X pivots to the back
            pivots_x = _first_nonzero_cols(matrix_x)
            other_x = [qq for qq in range(len(self)) if qq not in pivots_x]
            permutation = other_x + list(pivots_x)
            matrix_x = matrix_x[:, permutation]
            matrix_z = matrix_z[:, permutation]
            qudit_locs = qudit_locs[permutation]

            # row reduce and identify pivots in the Z sector, and move Z pivots to the back
            matrix_z = matrix_z.row_reduce()
            pivots_z = _first_nonzero_cols(matrix_z)
            other_z = [qq for qq in range(len(self)) if qq not in pivots_z]
            permutation = other_z + list(pivots_z)
            matrix_x = matrix_x[:, permutation]
            matrix_z = matrix_z[:, permutation]
            qudit_locs = qudit_locs[permutation]

            # some helpful numbers
            num_stabs_x = len(pivots_x)
            num_stabs_z = len(pivots_z)
            num_logicals = len(self) - num_stabs_x - num_stabs_z

            # row/column sectors of the parity check matrix
            rows_sx = slice(num_stabs_x)
            rows_sz = slice(num_stabs_z)
            cols_lx = cols_lz = slice(num_logicals)
            cols_sx = slice(cols_lx.stop, cols_lx.stop + num_stabs_x)
            cols_sz = slice(cols_sx.stop, cols_sx.stop + num_stabs_z)

            # fill in empty gauge sectors
            rows_gx = rows_gz = cols_gx = cols_gz = slice(0)

        else:
            # X-type and Z-type stabilizers in standard form
            stabilizers_x: npt.NDArray[np.int_] = self.get_stabilizer_ops(Pauli.X)
            stabilizers_z: npt.NDArray[np.int_] = self.get_stabilizer_ops(Pauli.Z)
            code = CSSCode(stabilizers_x, stabilizers_z, is_subsystem_code=False)
            (
                stabilizers_x,
                stabilizers_z,
                qudit_locs,
                (rows_sx, _, rows_sz, _),
                (cols_sx, _, _, cols_sz, _, cols_gk),
            ) = code.get_standard_form_data_xz()
            cols_gk = _join_slices(cols_gk)  # convert into indexable array

            # some helpful numbers
            num_stabs_x = len(stabilizers_x)
            num_stabs_z = len(stabilizers_z)
            num_gauges = self.gauge_dimension

            # canonicalized parity check matrices with qudits in the same order as above
            checks_x = self.canonicalized.matrix_x[:, qudit_locs]
            checks_z = self.canonicalized.matrix_z[:, qudit_locs]

            # row reduce X-type stabilizers + parity checks to ensure that gauge ops at the bottom
            permutation_x = _join_slices(cols_sx, cols_gk, cols_sz)
            matrix_x = np.vstack([stabilizers_x, checks_x])[:, permutation_x]  # type:ignore[assignment]
            matrix_x = ClassicalCode(matrix_x).canonicalized.matrix
            pivots_gx = _first_nonzero_cols(matrix_x)[num_stabs_x:] - num_stabs_x
            matrix_x = matrix_x[:, np.argsort(permutation_x)]

            # row reduce Z-type stabilizers + parity checks to ensure that gauge ops at the bottom
            permutation_z = _join_slices(cols_sz, cols_gk, cols_sx)
            matrix_z = np.vstack([stabilizers_z, checks_z])[:, permutation_z]  # type:ignore[assignment]
            matrix_z = ClassicalCode(matrix_z).canonicalized.matrix
            pivots_gz = _first_nonzero_cols(matrix_z)[num_stabs_z:] - num_stabs_z
            matrix_z = matrix_z[:, np.argsort(permutation_z)]

            """
            Row reducing the combiner stabilizer + gauge matrices added gauge ops to stabilizers to
            zero out entries above the gauge-pivot columns.  Remove the added gauge operators.
            """
            matrix_x[:num_stabs_x] += (
                stabilizers_x[:num_stabs_x, pivots_gx] @ matrix_x[num_stabs_x:]
            )
            matrix_z[:num_stabs_z] += (
                stabilizers_z[:num_stabs_z, pivots_gz] @ matrix_z[num_stabs_z:]
            )

            # identify row sectors for gauge ops
            rows_gx = slice(rows_sx.stop, rows_sx.stop + num_gauges)
            rows_gz = slice(rows_sz.stop, rows_sz.stop + num_gauges)

            # split logical vs. gauge column sectors
            cols_gx = cols_gk[pivots_gx]
            cols_gz = cols_gk[pivots_gz]
            cols_lx = [qq for qq in cols_gk if qq not in cols_gx]  # type:ignore[operator]
            cols_lz = [qq for qq in cols_gk if qq not in cols_gz]  # type:ignore[operator]

        return (
            matrix_x,
            matrix_z,
            qudit_locs,
            (rows_sx, rows_gx, rows_sz, rows_gz),
            (cols_sx, cols_gx, cols_lx, cols_sz, cols_gz, cols_lz),
        )

    def set_logical_ops_xz(
        self,
        logicals_x: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        logicals_z: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        *,
        validate: bool = True,
    ) -> None:
        """Set the logical operators of this code to the provided logical operators."""
        zero_block = np.zeros((self.dimension, len(self)), dtype=int)
        logical_ops = np.block(
            [
                [self.field(logicals_x), zero_block],
                [zero_block, self.field(logicals_z)],
            ]
        )
        self.set_logical_ops(logical_ops, validate=validate)

    def get_stabilizer_ops(
        self,
        pauli: PauliXZ | None = None,
        *,
        canonicalized: bool = False,
        symplectic: bool = False,
    ) -> galois.FieldArray:
        """Basis of stabilizer group generators for this code.

        If canonicalized is True, guarantee that the stabilizer matrix is canonicalized (i.e., row
        reduced) such that its rows are a minimal generating set for the stabilizer group.
        """
        if self._stabilizer_ops is None and self.is_subsystem_code:
            stabs_and_gauges_x = self.canonicalized.get_matrix(Pauli.X)
            stabs_and_gauges_z = self.canonicalized.get_matrix(Pauli.Z)
            stabs_and_logs_x = stabs_and_gauges_z.null_space()
            stabs_and_logs_z = stabs_and_gauges_x.null_space()
            stabs_and_gauges_and_logs_x = np.vstack([stabs_and_gauges_x, stabs_and_logs_x])
            stabs_and_gauges_and_logs_z = np.vstack([stabs_and_gauges_z, stabs_and_logs_z])
            assert isinstance(stabs_and_gauges_and_logs_x, galois.FieldArray)
            assert isinstance(stabs_and_gauges_and_logs_z, galois.FieldArray)

            stabs_x = stabs_and_gauges_and_logs_z.null_space()
            stabs_z = stabs_and_gauges_and_logs_x.null_space()
            self._stabilizer_ops = self.field(scipy.linalg.block_diag(stabs_x, stabs_z))

        stabilizer_ops = QuditCode.get_stabilizer_ops(self, pauli, canonicalized=canonicalized)
        if symplectic or pauli is None:
            return stabilizer_ops
        return stabilizer_ops.reshape(-1, 2, len(self))[:, pauli, :]  # type:ignore[return-value]

    def get_gauge_ops(
        self, pauli: PauliXZ | None = None, *, recompute: bool = False, symplectic: bool = False
    ) -> galois.FieldArray:
        """Basis of nontrivial logical Pauli operators for the gauge qudits of this code.

        Nontrivial logical Pauli operators for the gauge qudits are organized similarly to the
        logical Pauli operators computed by CSSCode.get_logical_ops.
        """
        gauge_ops = QuditCode.get_gauge_ops(self, pauli)
        if symplectic or pauli is None:
            return gauge_ops
        return gauge_ops.reshape(-1, 2, len(self))[:, pauli, :]  # type:ignore[return-value]

    def get_dual_subsystem_code(self) -> CSSCode:
        """Get the subsystem code that swaps gauge and logical qudits of this code."""
        matrix_x = np.vstack([self.get_stabilizer_ops(Pauli.X), self.get_logical_ops(Pauli.X)])
        matrix_z = np.vstack([self.get_stabilizer_ops(Pauli.Z), self.get_logical_ops(Pauli.Z)])
        return CSSCode(matrix_x, matrix_z, is_subsystem_code=self.dimension != 0)

    def get_distance(
        self, pauli: PauliXZ | None = None, *, bound: int | bool | None = None, **decoder_args: Any
    ) -> int | float:
        """Compute (or upper bound) the minimum weight of nontrivial logical operators.

        If `pauli is not None`, consider only `pauli`-type logical operators.

        If `bound is None`, compute an exact code distance by brute force.  Otherwise, compute
        upper bounds using a randomized algorithm and minimize over `bound` trials.  For a detailed
        explanation, see `get_one_distance_bound`.

        Additional arguments, if applicable, are passed to a decoder in `get_one_distance_bound`.
        """
        if not bound:
            return self.get_distance_exact(pauli)
        return self.get_distance_bound(num_trials=int(bound), pauli=pauli, **decoder_args)

    def get_distance_exact(self, pauli: PauliXZ | None = None) -> int | float:
        """Compute the minimum weight of nontrivial logical operators by brute force.

        If `pauli is not None`, consider only `pauli`-type logical operators.
        """
        if (known_distance := self.get_distance_if_known(pauli)) is not None:
            return known_distance

        if pauli is None:
            return min(self.get_distance_exact(Pauli.X), self.get_distance_exact(Pauli.Z))

        # we do not know the exact distance, so compute it
        logical_ops = self.get_logical_ops(pauli)
        stabilizer_ops = self.get_stabilizer_ops(pauli, canonicalized=True)
        if self.field.order == 2:
            distance = get_distance_quantum(
                logical_ops.view(np.ndarray).astype(np.uint8),
                stabilizer_ops.view(np.ndarray).astype(np.uint8),
                homogeneous=True,
            )
        else:
            warnings.warn(
                "Computing the exact distance of a non-binary code may take a (very) long time"
            )
            code_logical_ops = ClassicalCode.from_generator(logical_ops)
            code_stabilizer_ops = ClassicalCode.from_generator(stabilizer_ops)
            distance = min(
                np.count_nonzero(word_l + word_s)
                for word_l in code_logical_ops.iter_words(skip_zero=True)
                for word_s in code_stabilizer_ops.iter_words()
            )

        # save the exact distance and return
        if pauli is Pauli.X or self._equal_distance_xz:
            self._exact_distance_x = distance
        if pauli is Pauli.Z or self._equal_distance_xz:
            self._exact_distance_z = distance
        if self._exact_distance_x is not None and self._exact_distance_z is not None:
            self._exact_distance = min(self._exact_distance_x, self._exact_distance_z)
        return distance

    def get_distance_if_known(self, pauli: PauliXZ | None = None) -> int | float | None:
        """Retrieve exact distance, if known.  Otherwise return None."""
        assert pauli is None or pauli in PAULIS_XZ

        # the distances of dimension-0 codes are undefined
        if self.dimension == 0:
            self._exact_distance = self._exact_distance_x = self._exact_distance_z = np.nan

        if pauli is Pauli.X:
            return self._exact_distance_x
        elif pauli is Pauli.Z:
            return self._exact_distance_z
        return self._exact_distance

    def get_distance_bound(
        self,
        num_trials: int = 1,
        pauli: PauliXZ | None = None,
        *,
        cutoff: int | None = None,
        **decoder_args: Any,
    ) -> int | float:
        """Compute an upper bound on code distance by minimizing many individual upper bounds.

        If `pauli is not None`, consider only `pauli`-type logical operators.

        If passed a cutoff, don't bother trying to find distances less than the cutoff.

        Additional arguments, if applicable, are passed to a decoder in `get_one_distance_bound`.
        """
        if (known_distance := self.get_distance_if_known(pauli)) is not None:
            return known_distance

        min_bound = len(self)
        for _ in range(num_trials):
            if cutoff and min_bound <= cutoff:
                break
            new_bound = self.get_one_distance_bound(pauli, **decoder_args)
            min_bound = int(min(min_bound, new_bound))
        return min_bound

    def get_one_distance_bound(
        self,
        pauli: PauliXZ | None = None,
        **decoder_args: Any,
    ) -> int:
        """Use a randomized algorithm to compute a single upper bound on code distance.

        If `pauli is not None`, consider only `pauli`-type logical operators.

        Additional arguments, if applicable, are passed to a decoder.

        This method uses the randomized algorithm described in arXiv:2308.07915, and also below.

        For ease of language, we henceforth assume (without loss of generality) that we are
        computing an X-distance.

        Pick a random Z-type logical operator Z(w_z) whose support is indicated by the bistring w_z.
        We now wish to find a low-weight Pauli-X string X(w_x) that
            (a) has a trivial syndrome, and
            (b) anti-commutes with Z(w_z),
        which together would imply that X(w_x) is a nontrivial X-type logical operator.
        Mathematically, these conditions are equivalent to requiring that
            (a) H_z @ w_x = 0, and
            (b) w_z @ w_x = 1,
        where H_z is the parity check matrix of the Z-type subcode that witnesses X-type errors.

        Conditions (a) and (b) can be combined into the single block-matrix equation
            ⌈ H_z   ⌉         ⌈ 0 ⌉
            ⌊ w_z.T ⌋ @ w_x = ⌊ 1 ⌋,
        where the "0" on the top right is interpreted as a zero vector.  This equation can be solved
        by decoding the syndrome [ 0, 0, ..., 0, 1 ].T for the parity check matrix [ H_z.T, w_z ].T.

        We solve the above decoding problem with a decoder in `decode`.  If the decoder fails to
        find a solution, try again with a new random operator Z(w_z).  If the decoder succeeds in
        finding a solution w_x, this solution corresponds to a logical X-type operator X(w_x) -- and
        presumably one of low Hamming weight, since decoders try to find low-weight solutions.
        Return the Hamming weight |w_x|.
        """
        pauli = pauli or random.choice(PAULIS_XZ)
        assert pauli in PAULIS_XZ

        # define pauli_z and code_z as if we are computing X-distance
        pauli_z: Literal[Pauli.Z, Pauli.X] = Pauli.Z if pauli is Pauli.X else Pauli.X
        code_z = self.get_code(pauli_z)

        # construct the effective syndrome
        effective_syndrome = np.zeros(code_z.num_checks + 1, dtype=int)
        effective_syndrome[-1] = 1

        logical_op_found = False
        while not logical_op_found:
            # support of pauli string with a trivial syndrome
            word = self.get_random_logical_op(pauli_z, ensure_nontrivial=True)

            # support of a candidate pauli-type logical operator
            effective_check_matrix = np.vstack([code_z.matrix, word])
            candidate_logical_op = decoders.decode(
                effective_check_matrix, effective_syndrome, **decoder_args
            )

            # check whether decoding was successful
            actual_syndrome = effective_check_matrix @ self.field(candidate_logical_op)
            logical_op_found = np.array_equal(actual_syndrome, effective_syndrome)

        # return the Hamming weight of the logical operator
        return int(np.count_nonzero(candidate_logical_op))

    def get_random_logical_op(
        self, pauli: PauliXZ, *, ensure_nontrivial: bool = False, seed: int | None = None
    ) -> galois.FieldArray:
        """Return a random logical operator of a given type.

        A random logical operator may be trivial, which is to say that it may be equal to the
        identity modulo stabilizers.  If `ensure_nontrivial is True`, ensure that the logical
        operator we return is nontrivial.
        """
        assert pauli is Pauli.X or pauli is Pauli.Z
        logical_ops = self.get_logical_ops(pauli)
        stabilizer_ops = self.get_stabilizer_ops(pauli, canonicalized=True)
        ops = np.vstack([logical_ops, stabilizer_ops])
        random_array = get_random_array(
            self.field,
            len(ops),
            seed=seed,
            satisfy=lambda array: (np.any(array[: self.dimension]) if ensure_nontrivial else True),
        )
        return random_array @ ops

    def reduce_logical_op(self, pauli: PauliXZ, logical_index: int, **decoder_args: Any) -> None:
        """Reduce the weight of a logical operator.

        A minimal-weight logical operator is found by enforcing that it has a trivial syndrome, and
        that it commutes with all logical operators except its dual.  This is essentially the same
        method as that used in CSSCode.get_one_distance_bound.
        """
        assert pauli is Pauli.X or pauli is Pauli.Z
        assert 0 <= logical_index < self.dimension

        # effective check matrix = syndromes and dual-pauli logical operators
        code = self.get_code(~pauli)  # type:ignore[arg-type]
        dual_logical_ops = self.get_logical_ops(~pauli)  # type:ignore[arg-type]
        effective_check_matrix = np.vstack([code.matrix, dual_logical_ops])
        dual_op_index = code.num_checks + logical_index

        # enforce that the new logical operator commutes with everything except its dual
        effective_syndrome = np.zeros((code.num_checks + self.dimension), dtype=int)
        effective_syndrome[dual_op_index] = 1

        logical_op_found = False
        while not logical_op_found:
            candidate_logical_op = decoders.decode(
                effective_check_matrix, effective_syndrome, **decoder_args
            )
            actual_syndrome = effective_check_matrix @ self.field(candidate_logical_op)
            logical_op_found = np.array_equal(actual_syndrome, effective_syndrome)

        assert self._logical_ops is not None
        self._logical_ops.shape = (2, self.dimension, 2, len(self))
        self._logical_ops[pauli, logical_index, pauli, :] = candidate_logical_op
        self._logical_ops.shape = (2 * self.dimension, 2 * len(self))

    def reduce_logical_ops(self, pauli: PauliXZ | None = None, **decoder_args: Any) -> None:
        """Reduce the weight of all logical operators."""
        assert pauli is None or pauli in PAULIS_XZ
        if pauli is None:
            self.reduce_logical_ops(Pauli.X, **decoder_args)
            self.reduce_logical_ops(Pauli.Z, **decoder_args)
        else:
            for logical_index in range(self.dimension):
                self.reduce_logical_op(pauli, logical_index, **decoder_args)

    @staticmethod
    def stack(*codes: QuditCode, inherit_logicals: bool = True) -> CSSCode:
        """Stack the given CSS codes.

        The stacked code is obtained by having the input codes act on disjoint sets of bits.
        Stacking two codes with parameters [n_1, k_1, d_1] and [n_2, k_2, d_2], for example, results
        in a single code with parameters [n_1 + n_2, k_1 + k_2, min(d_1, d_2)].
        """
        if any(not isinstance(code, CSSCode) for code in codes):
            raise TypeError("CSSCode.stack requires CSSCode inputs")
        css_codes = cast(list[CSSCode], codes)
        code_x = ClassicalCode.stack(*[code.code_x for code in css_codes])
        code_z = ClassicalCode.stack(*[code.code_z for code in css_codes])
        code = CSSCode(
            code_x,
            code_z,
            is_subsystem_code=any(code.is_subsystem_code for code in codes),
            promise_equal_distance_xz=all(code._equal_distance_xz for code in css_codes),
        )
        if inherit_logicals:
            logicals_x = [code.get_logical_ops(Pauli.X) for code in css_codes]
            logicals_z = [code.get_logical_ops(Pauli.Z) for code in css_codes]
            code.set_logical_ops_xz(
                scipy.linalg.block_diag(*logicals_x),
                scipy.linalg.block_diag(*logicals_z),
            )
        return code

    @staticmethod
    def concatenate(
        inner: QuditCode,
        outer: QuditCode,
        outer_physical_to_inner_logical: Mapping[int, int] | Sequence[int] | None = None,
        *,
        inherit_logicals: bool = True,
    ) -> CSSCode:
        """Concatenate two CSS codes.

        The concatenated code uses the logical qudits of the "inner" code as the physical qudits of
        the "outer" code, with outer_physical_to_inner_logical defining the map from outer physical
        qudit index to inner logical qudit index.

        This method nominally assumes that len(outer_physical_to_inner_logical) is equal to both the
        number of logical qudits of the inner code and the number of physical qudits of the outer
        code.  If len(outer_physical_to_inner_logical) is larger than the number of inner logicals
        or outer physicals, then copies of the inner and outer codes are used (stacked together) to
        match the expected number of "intermediate" qudits.  If no outer_physical_to_inner_logical
        mapping is provided, then this method "interleaves" intermediate qubits, using each logical
        qubit of an inner block as a physical qubit of a different outer code block.

        If inherit_logicals is True, use the logical operators of the outer code as the logical
        operators of the concatenated code.  Otherwise, logical operators of the concatenated code
        get recomputed from scratch.
        """
        if not isinstance(inner, CSSCode) or not isinstance(outer, CSSCode):
            raise TypeError("CSSCode.concatenate requires CSSCode inputs")

        # stack copies of the inner and outer codes (if necessary) and permute inner logicals
        inner, outer = QuditCode._standardize_concatenation_inputs(
            inner, outer, outer_physical_to_inner_logical
        )
        assert isinstance(inner, CSSCode) and isinstance(outer, CSSCode)

        """
        Parity checks inherited from the outer code are nominally defined in terms of their support
        on logical operators of the inner code.  Expand these parity checks into their support on
        the physical qudits of the inner code.
        """
        outer_checks_x = outer.matrix_x @ inner.get_logical_ops(Pauli.X)
        outer_checks_z = outer.matrix_z @ inner.get_logical_ops(Pauli.Z)

        # combine parity checks of the inner and outer codes
        code = CSSCode(
            np.vstack([inner.matrix_x, outer_checks_x]),
            np.vstack([inner.matrix_z, outer_checks_z]),
        )

        if inherit_logicals:
            code._logical_ops = outer.get_logical_ops() @ inner.get_logical_ops()
        return code

    def get_logical_error_rate_func(
        self,
        num_samples: int,
        max_error_rate: float = 0.3,
        pauli_bias: Sequence[float] | None = None,
        **decoder_args: Any,
    ) -> Callable[[float | Sequence[float]], tuple[float, float]]:
        """Construct a function from physical --> logical error rate in a code capacity model.

        In addition to the logical error rate, the constructed function returns an uncertainty
        (standard error) in that logical error rate.

        The physical error rate provided to the constructed function is the probability with which
        each qubit experiences a Pauli error.  The constructed function will throw an error if
        given a physical error rate larger than max_error_rate.  If a pauli_bias is provided, it is
        treated as the relative probabilities of an X, Y, and Z error on each qubit; otherwise,
        these errors occur with equal probability, corresponding to a depolarizing error.

        The logical error rate returned by the constructed function the probability with which a
        code error (obtained by sampling independent errors on all qubits) is converted into a
        logical error by the decoder.

        See ClassicalCode.get_logical_error_rate_func for more details about how this method works.
        """
        # collect relative probabilities of Z, X, and Y errors
        pauli_bias_zxy: npt.NDArray[np.float_] | None
        if pauli_bias is not None:
            assert len(pauli_bias) == 3
            pauli_bias_zxy = np.array([pauli_bias[2], pauli_bias[0], pauli_bias[1]], dtype=float)
            pauli_bias_zxy /= np.sum(pauli_bias_zxy)
        else:
            pauli_bias_zxy = None

        stabilizer_ops_x = self.get_stabilizer_ops(Pauli.X, canonicalized=False)
        stabilizer_ops_z = self.get_stabilizer_ops(Pauli.Z, canonicalized=False)

        # construct decoders
        decoder_x = decoders.get_decoder(stabilizer_ops_z, **decoder_args)
        decoder_z = decoders.get_decoder(stabilizer_ops_x, **decoder_args)
        if not isinstance(decoder_x, decoders.DirectDecoder):
            decoder_x = decoders.DirectDecoder.from_indirect(decoder_x, stabilizer_ops_z)
        if not isinstance(decoder_z, decoders.DirectDecoder):
            decoder_z = decoders.DirectDecoder.from_indirect(decoder_z, stabilizer_ops_x)

        # identify logical operators
        logicals_x = self.get_logical_ops(Pauli.X)
        logicals_z = self.get_logical_ops(Pauli.Z)

        # compute decoding fidelities for each error weight
        sample_allocation = _get_sample_allocation(num_samples, len(self), max_error_rate)
        max_error_weight = len(sample_allocation) - 1
        fidelities = np.ones(max_error_weight + 1, dtype=float)
        variances = np.zeros(max_error_weight + 1, dtype=float)
        for weight in range(1, max_error_weight + 1):
            fidelities[weight], variances[weight] = self._estimate_decoding_fidelity_and_variance(
                weight,
                sample_allocation[weight],
                decoder_x,
                decoder_z,
                logicals_x,
                logicals_z,
                pauli_bias_zxy,
            )

        @np.vectorize
        def get_logical_error_rate(error_rate: float) -> tuple[float, float]:
            """Compute a logical error rate in a code-capacity model."""
            if error_rate > max_error_rate:
                raise ValueError(
                    "Cannot determine logical error rates for physical error rates greater than"
                    f" {max_error_rate}.  Try running get_logical_error_rate_func with a larger"
                    " max_error_rate."
                )
            probs = _get_error_probs_by_weight(len(self), error_rate, max_error_weight)
            return 1 - probs @ fidelities, np.sqrt(probs**2 @ variances)

        return get_logical_error_rate

    def _estimate_decoding_fidelity_and_variance(
        self,
        error_weight: int,
        num_samples: int,
        decoder_x: decoders.Decoder,
        decoder_z: decoders.Decoder,
        logicals_x: npt.NDArray[np.int_],
        logicals_z: npt.NDArray[np.int_],
        pauli_bias_zxy: npt.NDArray[np.float_] | None,
    ) -> tuple[float, float]:
        """Estimate a fidelity and its standard error when decoding a fixed number of errors."""
        num_failures = 0
        for _ in range(num_samples):
            # construct an error
            error_locations = np.random.choice(range(len(self)), size=error_weight, replace=False)
            error_paulis = np.random.choice([1, 2, 3], size=error_weight, p=pauli_bias_zxy)

            # decode Z-type errors
            error_locs_z = error_locations[(error_paulis % 2).astype(bool)]
            error_z = np.zeros(len(self), dtype=int)
            error_z[error_locs_z] = np.random.choice(
                range(1, self.field.order), size=len(error_locs_z)
            )
            residual_z = self.field(decoder_z.decode(error_z))
            if np.any(logicals_x @ residual_z):
                num_failures += 1
                continue

            # decode X-type errors
            error_locs_x = error_locations[error_paulis > 1]
            error_x = np.zeros(len(self), dtype=int)
            error_x[error_locs_x] = np.random.choice(
                range(1, self.field.order), size=len(error_locs_x)
            )
            residual_x = self.field(decoder_x.decode(error_x))
            if np.any(logicals_z @ residual_x):
                num_failures += 1

        infidelity = num_failures / num_samples
        variance = infidelity * (1 - infidelity) / num_samples
        return 1 - infidelity, variance


def _join_slices(*sectors: Slice) -> npt.NDArray[np.int_]:
    """Join index slices together into one slice."""
    return np.concatenate(
        [
            np.arange(sector.start or 0, sector.stop, sector.step or 1, dtype=int)
            if isinstance(sector, slice)
            else sector
            for sector in sectors
        ]
    )


def _first_nonzero_cols(matrix: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Get the first nonzero column for every row in a matrix."""
    if matrix.size == 0:
        return np.array([], dtype=int)
    boolean_matrix = matrix.reshape(matrix.shape[0], -1).view(np.ndarray).astype(bool)
    return np.argmax(boolean_matrix, axis=1)


def _get_sample_allocation(
    num_samples: int, block_length: int, max_error_rate: float
) -> npt.NDArray[np.int_]:
    """Construct an allocation of samples by error weight.

    This method returns an array whose k-th entry is the number of samples to devote to errors of
    weight k, given a maximum error rate that we care about.
    """
    probs = _get_error_probs_by_weight(block_length, max_error_rate)

    # zero out the distribution at k=0, flatten it out to the left of its peak, and renormalize
    probs[0] = 0
    probs[1 : np.argmax(probs)] = probs.max()
    probs /= np.sum(probs)

    # assign sample numbers according to the probability distribution constructed above,
    # increasing num_samples if necessary to deal with weird edge cases from round-off errors
    while np.sum(sample_allocation := np.round(probs * num_samples).astype(int)) < num_samples:
        num_samples += 1  # pragma: no cover

    # truncate trailing zeros and return
    nonzero = np.nonzero(sample_allocation)[0]
    return sample_allocation[: nonzero[-1] + 1]


def _get_error_probs_by_weight(
    block_length: int, error_rate: float, max_weight: int | None = None
) -> npt.NDArray[np.float_]:
    """Build an array whose k-th entry is the probability of a weight-k error in a code.

    If a code has block_length n and each bit has an independent probability p = error_rate of an
    error, then the probability of k errors is (n choose k) p**k (1-p)**(n-k).

    We compute the above probability using logarithms because otherwise the combinatorial factor
    (n choose k) might be too large to handle.
    """
    max_weight = max_weight or block_length

    # deal with some pathological cases
    if error_rate == 0:
        probs = np.zeros(max_weight + 1)
        probs[0] = 1
        return probs
    elif error_rate == 1:
        probs = np.zeros(max_weight + 1)
        probs[block_length:] = 1
        return probs

    log_error_rate = np.log(error_rate)
    log_one_minus_error_rate = np.log(1 - error_rate)
    log_probs = [
        _log_choose(block_length, kk)
        + kk * log_error_rate
        + (block_length - kk) * log_one_minus_error_rate
        for kk in range(max_weight + 1)
    ]
    return np.exp(log_probs)


@functools.cache
def _log_choose(n: int, k: int) -> float:
    """Natural logarithm of (n choose k) = Gamma(n+1) / ( Gamma(k+1) * Gamma(n-k+1) )."""
    return (
        scipy.special.gammaln(n + 1)
        - scipy.special.gammaln(k + 1)
        - scipy.special.gammaln(n - k + 1)
    )
