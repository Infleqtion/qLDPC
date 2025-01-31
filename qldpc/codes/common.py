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
import math
import random
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Iterator, Literal, cast

import galois
import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.special
import stim

from qldpc import abstract, decoders, external
from qldpc.abstract import DEFAULT_FIELD_ORDER
from qldpc.objects import PAULIS_XZ, Node, Pauli, PauliXZ, QuditOperator, conjugate_xz, op_to_string

from ._distance import get_distance_classical, get_distance_quantum


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
            self._matrix = self.field(matrix)

        if field is not None and field != self.field.order:
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

    @property
    def matrix(self) -> galois.FieldArray:
        """Parity check matrix of this code."""
        return self._matrix

    @property
    def num_checks(self) -> int:
        """Number of parity checks in this code."""
        return self.matrix.shape[0]

    @functools.cached_property
    def rank(self) -> int:
        """Rank of this code's parity check matrix.

        Equivalently, the number of linearly independent parity checks in this code.
        """
        matrix_rref = self.matrix.row_reduce()
        nonzero_rows = np.any(matrix_rref, axis=1)
        return np.count_nonzero(nonzero_rows)

    @property
    def dimension(self) -> int:
        """The number of logical (qu)dits encoded by this code."""
        return len(self) - self.rank

    @functools.cached_property
    def graph(self) -> nx.DiGraph:
        """Tanner graph of this code."""
        return self.matrix_to_graph(self.matrix)

    @abc.abstractmethod
    def __len__(self) -> int:
        """The block length of this code."""

    @staticmethod
    @abc.abstractmethod
    def matrix_to_graph(matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]]) -> nx.DiGraph:
        """Convert a parity check matrix into a Tanner graph."""

    @staticmethod
    @abc.abstractmethod
    def graph_to_matrix(graph: nx.DiGraph) -> galois.FieldArray:
        """Convert a Tanner graph into a parity check matrix."""

    @abc.abstractmethod
    def __str__(self) -> str:
        """Human-readable representation of this code."""


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
    _exact_distance: int | float | None = None

    def __eq__(self, other: object) -> bool:
        """Equality test between two code instances."""
        return (
            isinstance(other, ClassicalCode)
            and self._field is other._field
            and np.array_equal(self._matrix, other._matrix)
        )

    def __str__(self) -> str:
        """Human-readable representation of this code."""
        text = ""
        if self.field.order == 2:
            text += f"{self.name} on {self.num_bits} bits"
        else:
            text += f"{self.name} on {self.num_bits} symbols over {self.field_name}"
        text += f", with parity check matrix\n{self.matrix}"
        return text

    def __contains__(
        self, words: npt.NDArray[np.int_] | Sequence[int] | Sequence[Sequence[int]]
    ) -> bool:
        """Does this code contain the given word(s)?"""
        return not np.any(self.matrix @ self.field(words).T)

    @staticmethod
    def equiv(code_a: ClassicalCode, code_b: ClassicalCode) -> bool:
        """Are two classical codes equivalent?  That is, do they have the same code words?"""
        return code_a.field is code_b.field and np.array_equal(
            code_a.canonicalized().matrix, code_b.canonicalized().matrix
        )

    def canonicalized(self) -> ClassicalCode:
        """The same code with its parity matrix in reduced row echelon form."""
        rows = [row for row in self.matrix.row_reduce() if np.any(row)]
        return ClassicalCode(rows, self.field.order)

    @staticmethod
    def matrix_to_graph(matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]]) -> nx.DiGraph:
        """Convert a parity check matrix H into a Tanner graph.

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
        num_bits = sum(1 for node in graph.nodes() if node.is_data)
        num_checks = len(graph.nodes()) - num_bits
        field = getattr(graph, "order", DEFAULT_FIELD_ORDER)
        matrix = galois.GF(field).Zeros((num_checks, num_bits))
        for node_c, node_b, data in graph.edges(data=True):
            matrix[node_c.index, node_b.index] = data.get("val", 1)
        return matrix

    @functools.cached_property
    def generator(self) -> galois.FieldArray:
        """Generator of this code: a matrix whose rows for a basis for code words."""
        return self.matrix.null_space()

    def words(self) -> galois.FieldArray:
        """Code words of this code."""
        vectors = itertools.product(self.field.elements, repeat=self.generator.shape[0])
        return self.field(list(vectors)) @ self.generator

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

    def __len__(self) -> int:
        """The block length of this code."""
        return self._matrix.shape[1]

    @property
    def num_bits(self) -> int:
        """Number of data bits in this code."""
        return len(self)

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
        if (known_distance := self._get_distance_if_known(vector)) is not None:
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

    def _get_distance_if_known(
        self, vector: Sequence[int] | npt.NDArray[np.int_] | None
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
        if (known_distance := self._get_distance_if_known(vector)) is not None:
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
            _fix_decoder_args_for_nonbinary_fields(decoder_args, self.field)
            correction = decoders.decode(
                self.matrix, self.matrix @ self.field(vector), **decoder_args
            )
            return int(np.count_nonzero(correction))

        # effective syndrome: a trivial "actual" syndrome, and a nonzero overlap with a random word
        effective_syndrome = np.zeros(self.num_checks + 1, dtype=int)
        effective_syndrome[-1] = 1
        _fix_decoder_args_for_nonbinary_fields(decoder_args, self.field, bound_index=-1)

        valid_candidate_found = False
        while not valid_candidate_found:
            # construct an effective check matrix with a random nonzero word
            random_word = get_random_array(self.field, self.num_bits, satisfy=lambda vec: vec.any())
            effective_check_matrix = np.vstack([self.matrix, random_word]).view(np.ndarray)

            # find a low-weight candidate code word
            candidate = decoders.decode(effective_check_matrix, effective_syndrome, **decoder_args)

            # check whether we found a valid candidate
            # NOTE: we can mod out by the field order because non-prime fields aren't allowed here
            actual_syndrome = effective_check_matrix @ candidate % self.field.order
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
        return self.num_bits, self.dimension, distance

    def get_weight(self) -> int:
        """Compute the weight of the largest check."""
        return max(np.count_nonzero(row) for row in self.matrix)

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
        matrix = self.canonicalized().matrix.view(np.ndarray)
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
        assert all(0 <= bit < self.num_bits for bit in bits)
        bits_to_keep = [bit for bit in range(self.num_bits) if bit not in bits]
        new_generator = self.generator[:, bits_to_keep]
        return ClassicalCode.from_generator(new_generator, self.field.order)

    def shorten(self, *bits: int) -> ClassicalCode:
        """Shorten a code to the words that are zero on the specified bits, and delete those bits.

        To shorten a code on a given bit, we:
        - move the bit to the first position,
        - row-reduce the generator matrix into the form [ identity_matrix, other_stuff ], and
        - delete the first row and column from the generator matrix.
        """
        assert all(0 <= bit < self.num_bits for bit in bits)
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
        if self.field.order != 2:
            raise ValueError("Logical error rate calculations are only supported for binary codes")

        decoder = decoders.get_decoder(self.matrix, **decoder_args)

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
            error = self.field.Zeros(len(self))
            error[error_locations] = self.field(1)

            # decode the error
            correction = self.field(decoder.decode(self.matrix @ error))
            residual_error = error - correction
            if np.any(residual_error):
                num_failures += 1

        infidelity = num_failures / num_samples
        variance = infidelity * (1 - infidelity) / num_samples
        return 1 - infidelity, variance


################################################################################
# quantum codes


class QuditCode(AbstractCode):
    """Quantum stabilizer code for Galois qudits, with dimension q = p^m for prime p and integer m.

    The parity check matrix of a QuditCode has dimensions (num_checks, 2 * num_qudits), and can be
    written as a block matrix in the form H = [H_x|H_z].  Each block has num_qudits columns.

    The entries H_x[c, d] = r_x and H_z[c, d] = r_z indicate that check c addresses qudit d with the
    operator X(r_x) * Z(r_z), where r_x, r_z range over the base field, and X(r), Z(r) are
    generalized Pauli operators.  Specifically:
    - X(r) = sum_{j=0}^{q-1} |j+r><j| is a shift operator, and
    - Z(r) = sum_{j=0}^{q-1} w^{j r} |j><j| is a phase operator, with w = exp(2 pi i / q).

    Here j and r are not integers, but elements of the Galois field GF(q), which has different
    rules for addition and multiplication when q is not a prime number.

    Helpful lecture by Gottesman: https://www.youtube.com/watch?v=JWg4zrNAF-g
    """

    _matrix: galois.FieldArray
    _logical_ops: galois.FieldArray | None = None

    _exact_distance: int | float | None = None

    def __init__(
        self,
        matrix: AbstractCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | None = None,
        *,
        validate: bool = True,
    ) -> None:
        """Construct a qudit code from a parity check matrix over a finite field."""
        AbstractCode.__init__(self, matrix, field)
        if validate:
            assert not np.any(self.matrix @ conjugate_xz(self.matrix).T)

    def __eq__(self, other: object) -> bool:
        """Equality test between two code instances."""
        return (
            isinstance(other, QuditCode)
            and self._field is other._field
            and np.array_equal(self._matrix, other._matrix)
        )

    def __str__(self) -> str:
        """Human-readable representation of this code."""
        text = ""
        if self.field.order == 2:
            text += f"{self.name} on {self.num_qubits} qubits"
        else:
            text += f"{self.name} on {self.num_qudits} qudits over {self.field_name}"
        text += f", with parity check matrix\n{self.matrix}"
        return text

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
        """Compute the weight of the largest check."""
        matrix_x = self.matrix[:, : len(self)].view(np.ndarray)
        matrix_z = self.matrix[:, len(self) :].view(np.ndarray)
        matrix = matrix_x + matrix_z  # nonzero wherever a check addresses a qudit
        return max(np.count_nonzero(row) for row in matrix)

    @staticmethod
    def equiv(code_a: QuditCode, code_b: QuditCode) -> bool:
        """Are two quantum codes equivalent?  That is, do they have the same code space?"""
        return code_a.field is code_b.field and np.array_equal(
            code_a.canonicalized().matrix, code_b.canonicalized().matrix
        )

    def canonicalized(self, *, validate: bool = True) -> QuditCode:
        """The same code with its parity matrix in reduced row echelon form."""
        rows = [row for row in self.matrix.row_reduce() if np.any(row)]
        return QuditCode(rows, self.field.order, validate=validate)

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
        field = graph.order if hasattr(graph, "order") else DEFAULT_FIELD_ORDER
        return galois.GF(field)(matrix.reshape(num_checks, 2 * num_qudits))

    def get_stabilizers(self) -> list[str]:
        """Stabilizers (checks) of this code, represented by strings."""
        matrix = self.matrix.reshape(self.num_checks, 2, self.num_qudits)
        stabilizers = []
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
            stabilizers.append(" ".join(ops))
        return stabilizers

    @staticmethod
    def from_stabilizers(
        *stabilizers: str, field: int | None = None, validate: bool = True
    ) -> QuditCode:
        """Construct a QuditCode from the provided stabilizers."""
        field = field or DEFAULT_FIELD_ORDER
        check_ops = [stabilizer.split() for stabilizer in stabilizers]
        num_checks = len(check_ops)
        num_qudits = len(check_ops[0])
        operator: type[Pauli] | type[QuditOperator] = Pauli if field == 2 else QuditOperator

        matrix = np.zeros((num_checks, 2, num_qudits), dtype=int)
        for check, check_op in enumerate(check_ops):
            if len(check_op) != num_qudits:
                raise ValueError(f"Stabilizers 0 and {check} have different lengths")
            for qudit, op in enumerate(check_op):
                matrix[check, :, qudit] = operator.from_string(op).value

        return QuditCode(matrix.reshape(num_checks, 2 * num_qudits), field, validate=validate)

    def conjugated(
        self, qudits: slice | Sequence[int] | None = None, *, validate: bool = True
    ) -> QuditCode:
        """Apply local Fourier transforms to data qudits, swapping X-type and Z-type operators."""
        if qudits is None:
            qudits = self._default_conjugate if hasattr(self, "_default_conjugate") else ()
        matrix = self.matrix.copy().reshape(-1, 2, len(self))
        matrix[:, :, qudits] = matrix[:, ::-1, qudits]
        matrix = matrix.reshape(-1, 2 * len(self))
        code = QuditCode(matrix, field=self.field.order, validate=validate)

        if self._logical_ops is not None:
            logical_ops = self._logical_ops.copy().reshape(-1, 2, len(self))
            logical_ops[:, :, qudits] = logical_ops[:, ::-1, qudits]
            logical_ops = logical_ops.reshape(-1, 2 * len(self))
            code.set_logical_ops(logical_ops, validate=validate)

        return code

    def deformed(
        self, circuit: str | stim.Circuit, *, preserve_logicals: bool = False, validate: bool = True
    ) -> QuditCode:
        """Deform a code by the given circuit.

        If preserve_logicals==True, preserve the logical operators of the original code.
        """
        if not self.field.order == 2:
            raise ValueError("Code deformation is only supported for qubit codes")

        # convert the physical circuit into a tableau
        identity = stim.Circuit(f"I {len(self) - 1}")
        circuit = stim.Circuit(circuit) if isinstance(circuit, str) else circuit
        tableau = (circuit + identity).to_tableau()

        def deform_strings(strings: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            """Deform the given Pauli strings."""
            new_strings = []
            for check in strings:
                string = op_to_string(check)
                xs_zs = tableau(string).to_numpy()
                new_strings.append(np.concatenate(xs_zs))
            return self.field(new_strings)

        # deform this code by transforming its stabilizers
        matrix = deform_strings(self.matrix)
        new_code = QuditCode(matrix, validate=validate)

        # preserve or update logical operators, as applicable
        if preserve_logicals:
            new_code.set_logical_ops(self.get_logical_ops(), validate=validate)
        elif self._logical_ops is not None:
            logical_ops = deform_strings(self.get_logical_ops())
            new_code.set_logical_ops(logical_ops, validate=validate)

        return new_code

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
        return self.num_qudits, self.dimension, distance

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
        if (known_distance := self._get_distance_if_known()) is not None:
            return known_distance

        if self.field.order == 2:
            stabilizers = self.canonicalized().matrix
            logical_ops = self.get_logical_ops()
            distance = get_distance_quantum(
                logical_ops.view(np.ndarray).astype(np.uint8),
                stabilizers.view(np.ndarray).astype(np.uint8),
            )
        else:
            warnings.warn(
                "Computing the exact distance of a non-binary code may take a (very) long time"
            )
            distance = len(self)
            code_logical_ops = ClassicalCode.from_generator(self.get_logical_ops())
            code_stabilizers = ClassicalCode.from_generator(self.matrix)
            for word_l, word_s in itertools.product(
                code_logical_ops.iter_words(skip_zero=True),
                code_stabilizers.iter_words(),
            ):
                word = word_l + word_s
                support_x = word[: len(self)].view(np.ndarray)
                support_z = word[len(self) :].view(np.ndarray)
                support = support_x + support_z  # nonzero wherever a word addresses a qudit
                distance = min(distance, np.count_nonzero(support))

        self._exact_distance = int(distance)
        return distance

    def _get_distance_if_known(self) -> int | float | None:
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
        if (known_distance := self._get_distance_if_known()) is not None:
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

    def get_logical_ops(
        self, pauli: PauliXZ | None = None, *, recompute: bool = False
    ) -> galois.FieldArray:
        """Complete basis of nontrivial logical Pauli operators for this code.

        Logical operators are represented by a matrix logical_ops with shape (2 * k, 2 * n), where
        k and n are, respectively, the numbers of logical and physical qudits in this code.
        Each row of logical_ops is a vector that represents a logical operator.  The first
        (respectively, second) n entries of this vector indicate the support of *physical* X-type
        (respectively, Z-type) operators.  Similarly, the first (second) k rows correspond to
        *logical* X-type (Z-type) operators.  The logical operators at rows j and j+k are dual to
        each other, which is to say that the logical operator at row j commutes with the logical
        operators in all other rows except row j+k.

        If this method is passed a pauli operator (Pauli.X or Pauli.Z), it returns only the logical
        operators of that type.

        Due to the way that logical operators are constructed in this method, logical Z-type
        operators only address physical qudits by physical Z-type operators, while logical X-type
        operators address at least one physical qudits a physical X-type operator, but may
        additionally address physical qudits with physical Z-type operators.

        Logical operators are constructed using the method described in Section 4.1 of Gottesman's
        thesis (arXiv:9705052), slightly modified for qudits.
        """
        assert pauli is None or pauli in PAULIS_XZ

        # if requested, retrieve logical operators of one type only
        if pauli is not None:
            logical_ops = self.get_logical_ops(recompute=recompute).reshape(2, self.dimension, -1)
            return logical_ops[pauli, :, :]  # type:ignore[return-value]

        # memoize manually because other methods may modify the logical operators computed here
        if self._logical_ops is not None and not recompute:
            return self._logical_ops

        num_qudits = self.num_qudits
        num_checks = self.num_checks
        dimension = self.dimension
        identity = self.field.Identity(dimension)

        # keep track of current qudit locations
        qudit_locs = np.arange(num_qudits, dtype=int)

        # row reduce and identify pivots in the X sector
        matrix, pivots_x = _row_reduce(self.matrix)
        pivots_x = [pivot for pivot in pivots_x if pivot < num_qudits]
        other_x = [qq for qq in range(num_qudits) if qq not in pivots_x]

        # move the X pivots to the back
        matrix = matrix.reshape(num_checks * 2, num_qudits)
        matrix = np.hstack([matrix[:, other_x], matrix[:, pivots_x]])
        qudit_locs = np.hstack([qudit_locs[other_x], qudit_locs[pivots_x]])

        # row reduce and identify pivots in the Z sector
        matrix = matrix.reshape(num_checks, 2 * num_qudits)
        sub_matrix = matrix[len(pivots_x) :, num_qudits:]
        sub_matrix, pivots_z = _row_reduce(self.field(sub_matrix))
        matrix[len(pivots_x) :, num_qudits:] = sub_matrix
        other_z = [qq for qq in range(num_qudits) if qq not in pivots_z]

        # move the Z pivots to the back
        matrix = matrix.reshape(num_checks * 2, num_qudits)
        matrix = np.hstack([matrix[:, other_z], matrix[:, pivots_z]])
        qudit_locs = np.hstack([qudit_locs[other_z], qudit_locs[pivots_z]])

        # identify X-pivot and Z-pivot parity checks
        matrix = matrix.reshape(num_checks, 2 * num_qudits)[: len(pivots_x + pivots_z), :]
        checks_x = matrix[: len(pivots_x), :].reshape(len(pivots_x), 2, num_qudits)
        checks_z = matrix[len(pivots_x) :, :].reshape(len(pivots_z), 2, num_qudits)

        # run some sanity checks
        assert len(pivots_z) == 0 or pivots_z[-1] < num_qudits - len(pivots_x)
        assert dimension + len(pivots_x) + len(pivots_z) == num_qudits
        assert not np.any(checks_z[:, 0, :])

        # identify "sections" of columns / qudits
        section_k = slice(dimension)
        section_x = slice(dimension, dimension + len(pivots_x))
        section_z = slice(dimension + len(pivots_x), num_qudits)

        # construct X-pivot logical operators
        logicals_x = self.field.Zeros((dimension, 2, num_qudits))
        logicals_x[:, 0, section_k] = identity
        logicals_x[:, 0, section_z] = -checks_z[:, 1, :dimension].T
        logicals_x[:, 1, section_x] = -(
            checks_x[:, 1, section_z] @ checks_z[:, 1, section_k] + checks_x[:, 1, section_k]
        ).T

        # construct Z-pivot logical operators
        logicals_z = self.field.Zeros((dimension, 2, num_qudits))
        logicals_z[:, 1, section_k] = identity
        logicals_z[:, 1, section_x] = -checks_x[:, 0, :dimension].T

        # move qudits back to their original locations
        permutation = np.argsort(qudit_locs)
        logicals_x = logicals_x[:, :, permutation]
        logicals_z = logicals_z[:, :, permutation]

        # reshape and return
        logicals_x = logicals_x.reshape(dimension, 2 * num_qudits)
        logicals_z = logicals_z.reshape(dimension, 2 * num_qudits)
        self._logical_ops = self.field(np.vstack([logicals_x, logicals_z]))
        return self._logical_ops

    def set_logical_ops(
        self, logical_ops: npt.NDArray[np.int_] | Sequence[Sequence[int]], *, validate: bool = True
    ) -> None:
        """Set the logical operators of this code to the provided logical operators."""
        if validate:
            self.validate_candidate_logical_ops(logical_ops)
        self._logical_ops = self.field(logical_ops)

    def validate_candidate_logical_ops(
        self, logical_ops: npt.NDArray[np.int_] | Sequence[Sequence[int]]
    ) -> None:
        """Assert that the given logical operators are valid for this code."""
        logical_ops = self.field(logical_ops)

        logs_x = logical_ops[: self.dimension]
        logs_z = logical_ops[self.dimension :]
        inner_products = conjugate_xz(logs_x) @ logs_z.T
        if not np.array_equal(inner_products, np.eye(self.dimension, dtype=int)):
            raise ValueError("The given logical operators have incorrect commutation relations")

        if np.any(self.matrix @ conjugate_xz(logical_ops).T):
            raise ValueError("The given logical operators do not commute with stabilizers")

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
        code = QuditCode(matrix)
        if inherit_logicals:
            logicals_xx = [code.get_logical_ops(Pauli.X)[:, : len(code)] for code in codes]
            logicals_zx = [code.get_logical_ops(Pauli.Z)[:, : len(code)] for code in codes]
            logicals_xz = [code.get_logical_ops(Pauli.X)[:, len(code) :] for code in codes]
            logicals_zz = [code.get_logical_ops(Pauli.Z)[:, len(code) :] for code in codes]
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
        mapping is provided, then this method stacks the minimal number of inner and outer codes
        required make the number of inner logicals equal the number of outer physicals, and the k-th
        inner logical qudit is identified with the k-th outer physical qudit.

        If inherit_logicals is True, use the logical operators of the outer code as the logical
        operators of the concatenated code.  Otherwise, logical operators of the concatenated code
        get recomputed from scratch.
        """
        # stack copies of the inner and outer codes (if necessary) and permute inner logicals
        inner, outer = QuditCode._standardize_concatenation_inputs(
            inner, outer, outer_physical_to_inner_logical
        )

        """
        Parity checks inherited from the outer code are nominally defined in terms of their support
        on logical operators of the inner code.  Expand these parity checks into their support on
        the physical qudits of the inner code.
        """
        outer_checks = outer.matrix @ inner.get_logical_ops()

        # combine parity checks of the inner and outer codes
        code = QuditCode(np.vstack([inner.matrix, outer_checks]))

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
          If no outer_physical_to_inner_logical mapping is provided, then the k-th logical qudit of
          the inner code is used as the k-th physical qudit of the outer code.
        """
        if inner.field is not outer.field:
            raise ValueError("Cannot concatenate codes over different fields")

        # convert outer_physical_to_inner_logical into a tuple that we can use to permute an array
        if outer_physical_to_inner_logical is None:
            # default to the trivial mapping with the smallest possible number of qudits
            num_qudits = inner.dimension * len(outer) // math.gcd(inner.dimension, len(outer))
            outer_physical_to_inner_logical = tuple(range(num_qudits))
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
        inner._logical_ops = inner.field(
            inner.get_logical_ops()
            .reshape(2, inner.dimension, -1)[:, outer_physical_to_inner_logical, :]
            .reshape(2 * inner.dimension, -1)
        )

        return inner, outer


class CSSCode(QuditCode):
    """CSS qudit code, with separate X-type and Z-type parity checks.

    In order for the X-type and Z-type parity checks to be "compatible", the X-type stabilizers must
    commute with the Z-type stabilizers.  Mathematically, this requirement can be written as

    H_x @ H_z.T == 0,

    where H_x and H_z are, respectively, the parity check matrices of the classical codes that
    define the X-type and Z-type stabilizers of the CSS code.  Note that H_x witnesses Z-type errors
    and H_z witnesses X-type errors.

    The full parity check matrix of a CSSCode is
    ⌈  0 , H_z ⌉
    ⌊ H_x,  0  ⌋.
    """

    code_x: ClassicalCode  # X-type parity checks, measuring Z-type errors
    code_z: ClassicalCode  # Z-type parity checks, measuring X-type errors

    _exact_distance_x: int | float | None = None
    _exact_distance_z: int | float | None = None
    _balanced_codes: bool

    def __init__(
        self,
        code_x: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_z: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | None = None,
        *,
        promise_balanced_codes: bool = False,  # do the subcodes have the same parameters [n, k, d]?
        validate: bool = True,
    ) -> None:
        """Build a CSSCode from classical subcodes that specify X-type and Z-type parity checks."""
        self.code_x = ClassicalCode(code_x, field)
        self.code_z = ClassicalCode(code_z, field)

        if field is None and self.code_x.field is not self.code_z.field:
            raise ValueError("The sub-codes provided for this CSSCode are over different fields")
        self._field = self.code_x.field

        if validate:
            self._validate_subcodes()

        self._balanced_codes = promise_balanced_codes or self.code_x == self.code_z

    def _validate_subcodes(self) -> None:
        """Is this a valid CSS code?"""
        if not (
            self.code_x.num_bits == self.code_z.num_bits
            and not np.any(self.matrix_x @ self.matrix_z.T)
        ):
            raise ValueError("The sub-codes provided for this CSSCode are incompatible")

    def __eq__(self, other: object) -> bool:
        """Equality test between two code instances."""
        return (
            isinstance(other, type(self))
            and self._field is other._field
            and np.array_equal(self.code_x._matrix, other.code_x._matrix)
            and np.array_equal(self.code_z._matrix, other.code_z._matrix)
        )

    def __str__(self) -> str:
        """Human-readable representation of this code."""
        text = ""
        if self.field.order == 2:
            text += f"{self.name} on {self.num_qubits} qubits"
        else:
            text += f"{self.name} on {self.num_qudits} qudits over {self.field_name}"
        text += f"\nX-type parity checks:\n{self.matrix_x}"
        text += f"\nZ-type parity checks:\n{self.matrix_z}"
        return text

    @functools.cached_property
    def matrix(self) -> galois.FieldArray:
        """Overall parity check matrix."""
        matrix = np.block(
            [
                [np.zeros_like(self.matrix_z), self.matrix_z],
                [self.matrix_x, np.zeros_like(self.matrix_x)],
            ]
        )
        return self.field(matrix)

    @property
    def matrix_x(self) -> galois.FieldArray:
        """X-type parity checks."""
        return self.code_x.matrix

    @property
    def matrix_z(self) -> galois.FieldArray:
        """Z-type parity checks."""
        return self.code_z.matrix

    def canonicalized(self, *, validate: bool = True) -> CSSCode:
        """The same code with its parity matrix in reduced row echelon form."""
        return CSSCode(self.code_x.canonicalized(), self.code_z.canonicalized(), validate=validate)

    @property
    def num_checks_x(self) -> int:
        """Number of X-type parity checks in this code."""
        return self.matrix_x.shape[0]

    @property
    def num_checks_z(self) -> int:
        """Number of X-type parity checks in this code."""
        return self.matrix_z.shape[0]

    @property
    def num_checks(self) -> int:
        """Number of parity checks in this code."""
        return self.num_checks_x + self.num_checks_z

    def __len__(self) -> int:
        """Number of data qudits in this code."""
        return self.matrix_x.shape[1]

    @functools.cached_property
    def rank(self) -> int:
        """Rank of this code's parity check matrix.

        Equivalently, the number of linearly independent parity checks in this code.
        """
        return self.code_x.rank + self.code_z.rank

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
        if (known_distance := self._get_distance_if_known(pauli)) is not None:
            return known_distance

        if pauli is None:
            return min(self.get_distance_exact(Pauli.X), self.get_distance_exact(Pauli.Z))

        # we do not know the exact distance, so compute it
        if self.field.order == 2:
            code = self.code_x if pauli == Pauli.X else self.code_z
            stabilizers = code.canonicalized().matrix
            logical_ops = self.get_logical_ops(pauli).reshape(-1, 2, len(self))[:, pauli, :]
            distance = get_distance_quantum(
                logical_ops.view(np.ndarray).astype(np.uint8),
                stabilizers.view(np.ndarray).astype(np.uint8),
                homogeneous=True,
            )
        else:
            warnings.warn(
                "Computing the exact distance of a non-binary code may take a (very) long time"
            )
            logical_ops = self.get_logical_ops(pauli).reshape(-1, 2, len(self))[:, pauli, :]
            matrix = self.matrix_x if pauli == Pauli.X else self.matrix_z
            code_logical_ops = ClassicalCode.from_generator(logical_ops)
            code_stabilizers = ClassicalCode.from_generator(matrix)
            distance = min(
                np.count_nonzero(word_l + word_s)
                for word_l in code_logical_ops.iter_words(skip_zero=True)
                for word_s in code_stabilizers.iter_words()
            )

        # save the exact distance and return
        if pauli == Pauli.X or self._balanced_codes:
            self._exact_distance_x = distance
        if pauli == Pauli.Z or self._balanced_codes:
            self._exact_distance_z = distance
        return distance

    def _get_distance_if_known(self, pauli: PauliXZ | None = None) -> int | float | None:
        """Retrieve exact distance, if known.  Otherwise return None."""
        assert pauli is None or pauli in PAULIS_XZ

        # the distances of dimension-0 codes are undefined
        if self.dimension == 0:
            self._exact_distance_x = self._exact_distance_z = np.nan

        if pauli == Pauli.X:
            return self._exact_distance_x
        elif pauli == Pauli.Z:
            return self._exact_distance_z
        return (
            min(self._exact_distance_x, self._exact_distance_z)
            if self._exact_distance_x is not None and self._exact_distance_z is not None
            else None
        )

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
        if (known_distance := self._get_distance_if_known(pauli)) is not None:
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

        # define code_z and pauli_z as if we are computing X-distance
        code_z = self.code_z if pauli == Pauli.X else self.code_x
        pauli_z: Literal[Pauli.Z, Pauli.X] = Pauli.Z if pauli == Pauli.X else Pauli.X

        # construct the effective syndrome
        effective_syndrome = np.zeros(code_z.num_checks + 1, dtype=int)
        effective_syndrome[-1] = 1
        _fix_decoder_args_for_nonbinary_fields(decoder_args, self.field, bound_index=-1)

        logical_op_found = False
        while not logical_op_found:
            # support of pauli string with a trivial syndrome
            word = self.get_random_logical_op(pauli_z, ensure_nontrivial=True)

            # support of a candidate pauli-type logical operator
            effective_check_matrix = np.vstack([code_z.matrix, word]).view(np.ndarray)
            candidate_logical_op = decoders.decode(
                effective_check_matrix, effective_syndrome, **decoder_args
            )

            # check whether decoding was successful
            # NOTE: we can mod out by the field order because non-prime fields aren't allowed here
            actual_syndrome = effective_check_matrix @ candidate_logical_op % self.field.order
            logical_op_found = np.array_equal(actual_syndrome, effective_syndrome)

        # return the Hamming weight of the logical operator
        return int(np.count_nonzero(candidate_logical_op))

    def get_logical_ops(
        self, pauli: PauliXZ | None = None, *, recompute: bool = False
    ) -> galois.FieldArray:
        """Complete basis of nontrivial logical Pauli operators for this code.

        Logical operators are represented by a matrix logical_ops with shape (2 * k, 2 * n), where
        k and n are, respectively, the numbers of logical and physical qudits in this code.
        Each row of logical_ops is a vector that represents a logical operator.  The first
        (respectively, second) n entries of this vector indicate the support of *physical* X-type
        (respectively, Z-type) operators.  Similarly, the first (second) k rows correspond to
        *logical* X-type (Z-type) operators.  The logical operators at rows j and j+k are dual to
        each other, which is to say that the logical operator at row j commutes with the logical
        operators in all other rows except row j+k.

        If this method is passed a pauli operator (Pauli.X or Pauli.Z), it returns only the logical
        operators of that type.

        Logical X-type operators only address physical qudits by physical X-type operators, and
        logical Z-type operators only address physical qudits by physical Z-type operators.

        Logical operators are constructed using the method described in Section 4.1 of Gottesman's
        thesis (arXiv:9705052), slightly modified for qudits and CSSCodes.
        """
        assert pauli is None or pauli in PAULIS_XZ

        # if requested, retrieve logical operators of one type only
        if pauli is not None:
            logical_ops = self.get_logical_ops(recompute=recompute).reshape(2, self.dimension, -1)
            return logical_ops[pauli, :, :]  # type:ignore[return-value]

        # memoize manually because other methods may modify the logical operators computed here
        if self._logical_ops is not None and not recompute:
            return self._logical_ops

        num_qudits = self.num_qudits
        dimension = self.dimension
        identity = self.field.Identity(dimension)

        # identify check matrices for X/Z-type parity checks, and the current qudit locations
        checks_x: npt.NDArray[np.int_] = self.matrix_x
        checks_z: npt.NDArray[np.int_] = self.matrix_z
        qudit_locs = np.arange(num_qudits, dtype=int)

        # row reduce the check matrix for X-type errors and move its pivots to the back
        checks_x, pivots_x = _row_reduce(self.field(checks_x))
        other_x = [qq for qq in range(num_qudits) if qq not in pivots_x]
        checks_x = np.hstack([checks_x[:, other_x], checks_x[:, pivots_x]])
        checks_z = np.hstack([checks_z[:, other_x], checks_z[:, pivots_x]])
        qudit_locs = np.hstack([qudit_locs[other_x], qudit_locs[pivots_x]])

        # row reduce the check matrix for Z-type errors and move its pivots to the back
        checks_z, pivots_z = _row_reduce(self.field(checks_z))
        other_z = [qq for qq in range(num_qudits) if qq not in pivots_z]
        checks_x = np.hstack([checks_x[:, other_z], checks_x[:, pivots_z]])
        checks_z = np.hstack([checks_z[:, other_z], checks_z[:, pivots_z]])
        qudit_locs = np.hstack([qudit_locs[other_z], qudit_locs[pivots_z]])

        # run some sanity checks
        assert pivots_z[-1] < num_qudits - len(pivots_x)
        assert dimension + len(pivots_x) + len(pivots_z) == num_qudits

        # identify "sections" of columns / qudits
        section_k = slice(dimension)
        section_x = slice(dimension, dimension + len(pivots_x))
        section_z = slice(dimension + len(pivots_x), num_qudits)

        # construct logical X operators
        logicals_x = self.field.Zeros((dimension, num_qudits))
        logicals_x[:, section_k] = identity
        logicals_x[:, section_z] = -checks_z[: len(pivots_z), :dimension].T

        # construct logical Z operators
        logicals_z = self.field.Zeros((dimension, num_qudits))
        logicals_z[:, section_k] = identity
        logicals_z[:, section_x] = -checks_x[: len(pivots_x), :dimension].T

        # move qudits back to their original locations
        permutation = np.argsort(qudit_locs)
        logicals_x = logicals_x[:, permutation]
        logicals_z = logicals_z[:, permutation]

        self._logical_ops = self.field(scipy.linalg.block_diag(logicals_x, logicals_z))
        return self._logical_ops

    def set_logical_ops_xz(
        self,
        logicals_x: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        logicals_z: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        *,
        validate: bool = True,
    ) -> None:
        """Set the logical operators of this code to the provided logical operators."""
        logical_ops = scipy.linalg.block_diag(self.field(logicals_x), self.field(logicals_z))
        self.set_logical_ops(logical_ops, validate=validate)

    def get_random_logical_op(
        self, pauli: PauliXZ, *, ensure_nontrivial: bool = False, seed: int | None = None
    ) -> galois.FieldArray:
        """Return a random logical operator of a given type.

        A random logical operator may be trivial, which is to say that it may be equal to the
        identity modulo stabilizers.  If `ensure_nontrivial is True`, ensure that the logical
        operator we return is nontrivial.
        """
        assert pauli == Pauli.X or pauli == Pauli.Z
        if not ensure_nontrivial:
            return (self.code_z if pauli == Pauli.X else self.code_x).get_random_word(seed=seed)

        # generate random logical ops until we find ones with a nontrivial commutation relation
        noncommuting_ops_found = False
        while not noncommuting_ops_found:
            op_a = self.get_random_logical_op(pauli, ensure_nontrivial=False, seed=seed)
            op_b = self.get_random_logical_op(
                ~pauli,  # type:ignore[arg-type]
                ensure_nontrivial=False,
                seed=seed + 1 if seed is not None else None,
            )
            seed = seed + 2 if seed is not None else None
            noncommuting_ops_found = bool(np.any(op_a @ op_b))

        return op_a

    def reduce_logical_op(self, pauli: PauliXZ, logical_index: int, **decoder_args: Any) -> None:
        """Reduce the weight of a logical operator.

        A minimal-weight logical operator is found by enforcing that it has a trivial syndrome, and
        that it commutes with all logical operators except its dual.  This is essentially the same
        method as that used in CSSCode.get_one_distance_bound.
        """
        assert pauli == Pauli.X or pauli == Pauli.Z
        assert 0 <= logical_index < self.dimension

        dual_pauli = ~pauli
        assert dual_pauli == Pauli.X or dual_pauli == Pauli.Z

        # effective check matrix = syndromes and other logical operators
        if pauli == Pauli.X:
            code = self.code_z
            nonzero_dual_section = slice(self.num_qudits, 2 * self.num_qudits)
        else:
            code = self.code_x
            nonzero_dual_section = slice(self.num_qudits)
        all_dual_ops = self.get_logical_ops(dual_pauli)[:, nonzero_dual_section]
        effective_check_matrix = np.vstack([code.matrix, all_dual_ops]).view(np.ndarray)
        dual_op_index = code.num_checks + logical_index

        # enforce that the new logical operator commutes with everything except its dual
        effective_syndrome = np.zeros((code.num_checks + self.dimension), dtype=int)
        effective_syndrome[dual_op_index] = 1
        _fix_decoder_args_for_nonbinary_fields(decoder_args, self.field, bound_index=dual_op_index)

        logical_op_found = False
        while not logical_op_found:
            candidate_logical_op = decoders.decode(
                effective_check_matrix, effective_syndrome, **decoder_args
            )
            actual_syndrome = effective_check_matrix @ candidate_logical_op % self.field.order
            logical_op_found = np.array_equal(actual_syndrome, effective_syndrome)

        assert self._logical_ops is not None
        self._logical_ops.shape = (2, self.dimension, 2, self.num_qudits)
        self._logical_ops[pauli, logical_index, pauli, :] = candidate_logical_op
        self._logical_ops.shape = (2 * self.dimension, 2 * self.num_qudits)

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
            promise_balanced_codes=all(code._balanced_codes for code in css_codes),
            validate=False,
        )
        if inherit_logicals:
            logicals_x = [code.get_logical_ops(Pauli.X)[:, : len(code)] for code in css_codes]
            logicals_z = [code.get_logical_ops(Pauli.Z)[:, len(code) :] for code in css_codes]
            code.set_logical_ops_xz(
                scipy.linalg.block_diag(*logicals_x), scipy.linalg.block_diag(*logicals_z)
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
        mapping is provided, then this method stacks the minimal number of inner and outer codes
        required make the number of inner logicals equal the number of outer physicals, and the k-th
        inner logical qudit is identified with the k-th outer physical qudit.

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
        outer_checks_x = outer.matrix_x @ inner.get_logical_ops(Pauli.X)[:, : len(inner)]
        outer_checks_z = outer.matrix_z @ inner.get_logical_ops(Pauli.Z)[:, len(inner) :]

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
        if self.field.order != 2:
            raise ValueError("Logical error rate calculations are only supported for binary codes")

        # collect relative probabilities of Z, X, and Y errors
        pauli_bias_zxy: npt.NDArray[np.float_] | None
        if pauli_bias is not None:
            assert len(pauli_bias) == 3
            pauli_bias_zxy = np.array([pauli_bias[2], pauli_bias[0], pauli_bias[1]], dtype=float)
            pauli_bias_zxy /= np.sum(pauli_bias_zxy)
        else:
            pauli_bias_zxy = None

        # construct decoders and identify logical operators
        decoder_x = decoders.get_decoder(self.matrix_z, **decoder_args)
        decoder_z = decoders.get_decoder(self.matrix_x, **decoder_args)
        logicals_x = self.get_logical_ops(Pauli.X)[:, : len(self)]
        logicals_z = self.get_logical_ops(Pauli.Z)[:, len(self) :]

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
            error_locations = random.sample(range(len(self)), error_weight)
            pauli_errors = np.random.choice([1, 2, 3], size=error_weight, p=pauli_bias_zxy)
            error = np.zeros(len(self), dtype=int)
            error[error_locations] = pauli_errors

            # decode Z-type errors
            error_z = self.field(error % 2)
            correction_z = self.field(decoder_z.decode(self.matrix_x @ error_z))
            residual_z = error_z - correction_z
            if np.any(logicals_x @ residual_z):
                num_failures += 1
                continue

            # decode X-type errors
            error_x = self.field((error > 1).astype(int))
            correction_x = self.field(decoder_x.decode(self.matrix_z @ error_x))
            residual_x = error_x - correction_x
            if np.any(logicals_z @ residual_x):
                num_failures += 1

        infidelity = num_failures / num_samples
        variance = infidelity * (1 - infidelity) / num_samples
        return 1 - infidelity, variance


def _fix_decoder_args_for_nonbinary_fields(
    decoder_args: dict[str, object], field: type[galois.FieldArray], bound_index: int | None = None
) -> None:
    """Fix decoder arguments for nonbinary number fields.

    If the field has order greater than 2, then we can only decode
    (a) prime number fields, with
    (b) an integer-linear program decoder.

    If provided a bound_index, treat the constraint corresponding to this row of the parity check
    matrix as a lower bound (>=) rather than a strict equality (==) constraint.
    """
    if field.order > 2:
        if field.degree > 1:
            raise ValueError("Method only supported for prime number fields")
        decoder_args["with_ILP"] = True
        decoder_args["modulus"] = field.order
        if bound_index is not None:
            decoder_args["lower_bound_row"] = bound_index


def _row_reduce(matrix: galois.FieldArray) -> tuple[npt.NDArray[np.int_], list[int]]:
    """Perform Gaussian elimination on a matrix.

    Returns:
        matrix_rref: the reduced row echelon form of the matrix.
        pivot: the "pivot" columns of the reduced matrix.

    In reduced row echelon form, the first nonzero entry of each row is a 1, and these 1s
    occur at a unique columns for each row; these columns are the "pivots" of matrix_rref.
    """
    matrix_rref = matrix.row_reduce()
    pivots = [int(np.argmax(row != 0)) for row in matrix_rref if np.any(row)]
    return matrix_rref, pivots


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
