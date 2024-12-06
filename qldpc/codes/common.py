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
from collections.abc import Callable, Sequence
from typing import Any, Iterator, Literal

import galois
import networkx as nx
import numpy as np
import numpy.typing as npt

from qldpc import abstract, decoder, external
from qldpc.abstract import DEFAULT_FIELD_ORDER
from qldpc.objects import PAULIS_XZ, Node, Pauli, PauliXZ, QuditOperator


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

    @classmethod
    @abc.abstractmethod
    def matrix_to_graph(cls, matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]]) -> nx.DiGraph:
        """Convert a parity check matrix into a Tanner graph."""

    @classmethod
    @abc.abstractmethod
    def graph_to_matrix(cls, graph: nx.DiGraph) -> galois.FieldArray:
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

    @classmethod
    def equiv(cls, code_a: ClassicalCode, code_b: ClassicalCode) -> bool:
        """Are two classical codes equivalent?  That is, do they have the same code words?"""
        return code_a.field is code_b.field and np.array_equal(
            code_a.canonicalized().matrix, code_b.canonicalized().matrix
        )

    def canonicalized(self) -> ClassicalCode:
        """The same code with its parity matrix in reduced row echelon form."""
        rows = [row for row in self.matrix.row_reduce() if np.any(row)]
        return ClassicalCode(rows, self.field.order)

    @classmethod
    def matrix_to_graph(cls, matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]]) -> nx.DiGraph:
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

    @classmethod
    def graph_to_matrix(cls, graph: nx.DiGraph) -> galois.FieldArray:
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

    @classmethod
    def tensor_product(cls, code_a: ClassicalCode, code_b: ClassicalCode) -> ClassicalCode:
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

        self._exact_distance = min(
            np.count_nonzero(word) for word in self.iter_words(skip_zero=True)
        )
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
        vector: Sequence[int] | npt.NDArray[np.int_] | None = None,
        **decoder_args: Any,
    ) -> int | float:
        """Compute an upper bound on code distance by minimizing many individual upper bounds.

        If passed a vector, compute the minimum Hamming distance between the vector and a code word.

        Additional arguments, if applicable, are passed to a decoder in `get_one_distance_bound`.
        """
        if (known_distance := self._get_distance_if_known(vector)) is not None:
            return known_distance

        return min(
            (self.get_one_distance_bound(vector=vector, **decoder_args) for _ in range(num_trials)),
            default=self.num_bits,
        )

    def get_one_distance_bound(
        self, *, vector: Sequence[int] | npt.NDArray[np.int_] | None = None, **decoder_args: Any
    ) -> int | float:
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
            correction = decoder.decode(
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
            candidate = decoder.decode(effective_check_matrix, effective_syndrome, **decoder_args)

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

    @classmethod
    def random(
        cls, bits: int, checks: int, field: int | None = None, *, seed: int | None = None
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

    @classmethod
    def from_generator(
        self, generator: npt.NDArray[np.int_] | Sequence[Sequence[int]], field: int | None = None
    ) -> ClassicalCode:
        """Construct a ClassicalCode from a generator matrix."""
        return ~ClassicalCode(generator, field)

    @classmethod
    def from_name(cls, name: str) -> ClassicalCode:
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
        matrix = np.array([row for row in self.canonicalized().matrix])
        checks_str = ["[" + ",".join(map(str, line)) + "]" for line in matrix]
        matrix_str = "[" + ",".join(checks_str) + "]"
        code_str = f"CheckMatCode({matrix_str}, GF({self.field.order}))"
        group_str = "AutomorphismGroup" if self.field.order == 2 else "PermutationAutomorphismGroup"
        return abstract.Group.from_name(f"{group_str}({code_str})", field=self.field.order)

    @classmethod
    def stack(cls, code_a: ClassicalCode, code_b: ClassicalCode) -> ClassicalCode:
        """Stack two classical codes.

        The stacked code is obtained by having the input codes act on disjoint sets of bits.
        Stacking two codes with parameters [n_1, k_1, d_1] and [n_2, k_2, d_2] results in a single
        code with parameters [n_1 + n_2, k_1 + k_2, min(d_1, d_2)].
        """
        if code_a.field is not code_b.field:
            raise ValueError("Cannot join codes over different fields")
        block_matrix = [
            [code_a.matrix, np.zeros((code_a.num_checks, len(code_b)), dtype=int)],
            [np.zeros((code_b.num_checks, len(code_a)), dtype=int), code_b.matrix],
        ]
        return ClassicalCode(np.block(block_matrix), field=code_a.field.order)

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


################################################################################
# quantum codes


class QuditCode(AbstractCode):
    """Quantum stabilizer code for Galois qudits, with dimension q = p^m for prime p and integer m.

    The parity check matrix of a QuditCode has dimensions (num_checks, 2 * num_qudits), and can be
    written as a block matrix in the form H = [H_z|H_x].  Each block has num_qudits columns.

    The entries H_x[c, d] = r_x and H_z[c, d] = r_z iff check c addresses qudit d with the operator
    X(r_x) * Z(r_z), where r_x, r_z range over the base field, and X(r), Z(r) are generalized Pauli
    operators.  Specifically:
    - X(r) = sum_{j=0}^{q-1} |j+r><j| is a shift operator, and
    - Z(r) = sum_{j=0}^{q-1} w^{j r} |j><j| is a phase operator, with w = exp(2 pi i / q).

    Warning: here j, r, s, etc. not integers, but elements of the Galois field GF(q), which has
    different rules for addition and multiplication when q is not a prime number.

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
        skip_validation: bool = False,
    ) -> None:
        """Construct a qudit code from a parity check matrix over a finite field."""
        AbstractCode.__init__(self, matrix, field)
        if not skip_validation:
            shape_xz = (self.num_checks, 2, -1)
            matrix_xz = self.matrix.reshape(shape_xz)[:, ::-1, :].reshape(self.matrix.shape)
            assert not np.any(self.matrix @ matrix_xz.T)

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
        matrix_x = self.matrix[:, len(self) :].view(np.ndarray)
        matrix_z = self.matrix[:, : len(self)].view(np.ndarray)
        matrix = matrix_x + matrix_z  # nonzero wherever a check addresses a qudit
        return max(np.count_nonzero(row) for row in matrix)

    @classmethod
    def equiv(cls, code_a: QuditCode, code_b: QuditCode) -> bool:
        """Are two quantum codes equivalent?  That is, do they have the same code space?"""
        return code_a.field is code_b.field and np.array_equal(
            code_a.canonicalized().matrix, code_b.canonicalized().matrix
        )

    def canonicalized(self) -> QuditCode:
        """The same code with its parity matrix in reduced row echelon form."""
        rows = [row for row in self.matrix.row_reduce() if np.any(row)]
        return QuditCode(rows, self.field.order)

    @classmethod
    def matrix_to_graph(cls, matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]]) -> nx.DiGraph:
        """Convert a parity check matrix into a Tanner graph."""
        graph = nx.DiGraph()
        matrix = np.reshape(matrix, (len(matrix), 2, -1))
        for row, col_zx, col in zip(*np.nonzero(matrix)):
            node_check = Node(index=int(row), is_data=False)
            node_qudit = Node(index=int(col), is_data=True)
            graph.add_edge(node_check, node_qudit)

            qudit_op = graph[node_check][node_qudit].get(QuditOperator, QuditOperator())
            vals_xz = list(qudit_op.value)
            vals_xz[1 - col_zx] += int(matrix[row, col_zx, col])
            graph[node_check][node_qudit][QuditOperator] = QuditOperator(tuple(vals_xz))

        # remember order of the field, and use Pauli operators if appropriate
        if isinstance(matrix, galois.FieldArray):
            graph.order = type(matrix).order
            if graph.order == 2:
                for _, __, data in graph.edges(data=True):
                    data[Pauli] = Pauli(data[QuditOperator].value)
                    del data[QuditOperator]

        return graph

    @classmethod
    def graph_to_matrix(cls, graph: nx.DiGraph) -> galois.FieldArray:
        """Convert a Tanner graph into a parity check matrix."""
        num_qudits = sum(1 for node in graph.nodes() if node.is_data)
        num_checks = len(graph.nodes()) - num_qudits
        matrix = np.zeros((num_checks, 2, num_qudits), dtype=int)
        for node_check, node_qudit, data in graph.edges(data=True):
            op = data.get(QuditOperator) or data.get(Pauli)
            matrix[node_check.index, :, node_qudit.index] = op.value[::-1]
        field = graph.order if hasattr(graph, "order") else DEFAULT_FIELD_ORDER
        return galois.GF(field)(matrix.reshape(num_checks, 2 * num_qudits))

    def get_stabilizers(self) -> list[str]:
        """Stabilizers (checks) of this code, represented by strings."""
        matrix = self.matrix.reshape(self.num_checks, 2, self.num_qudits)
        stabilizers = []
        for check in range(self.num_checks):
            ops = []
            for qudit in range(self.num_qudits):
                val_x = matrix[check, ~Pauli.X, qudit]
                val_z = matrix[check, ~Pauli.Z, qudit]
                vals_xz = (val_x, val_z)
                if self.field.order == 2:
                    ops.append(str(Pauli(vals_xz)))
                else:
                    ops.append(str(QuditOperator(vals_xz)))
            stabilizers.append(" ".join(ops))
        return stabilizers

    @classmethod
    def from_stabilizers(
        cls, *stabilizers: str, field: int | None = None, skip_validation: bool = False
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
                matrix[check, :, qudit] = operator.from_string(op).value[::-1]

        return QuditCode(
            matrix.reshape(num_checks, 2 * num_qudits), field, skip_validation=skip_validation
        )

    def conjugated(
        self, qudits: slice | Sequence[int] | None = None, *, skip_validation: bool = False
    ) -> QuditCode:
        """Apply local Fourier transforms to data qudits, swapping X-type and Z-type operators."""
        if qudits is None:
            qudits = self._default_conjugate if hasattr(self, "_default_conjugate") else ()
        num_checks = len(self.matrix)
        matrix = np.reshape(self.matrix.copy(), (num_checks, 2, -1))
        matrix[:, :, qudits] = matrix[:, ::-1, qudits]
        return QuditCode(matrix.reshape(num_checks, -1), skip_validation=skip_validation)

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
        distance = self.get_distance(bound=bound, vector=None, **decoder_args)
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

        minimum_weight = len(self)
        for word in ClassicalCode(self.matrix).iter_words(skip_zero=True):
            support_x = word[: len(self)].view(np.ndarray)
            support_z = word[len(self) :].view(np.ndarray)
            support = support_x + support_z  # nonzero wherever a word addresses a qudit
            minimum_weight = min(minimum_weight, np.count_nonzero(support))

        self._exact_distance = minimum_weight
        return minimum_weight

    def _get_distance_if_known(self) -> int | float | None:
        """Retrieve exact distance, if known.  Otherwise return None."""
        # the distance of dimension-0 codes is undefined
        if self.dimension == 0:
            self._exact_distance = np.nan

        return self._exact_distance

    def get_distance_bound(self, num_trials: int = 1, **decoder_args: Any) -> int | float:
        """Compute an upper bound on code distance by minimizing many individual upper bounds.

        Additional arguments, if applicable, are passed to a decoder in `get_one_distance_bound`.
        """
        if (known_distance := self._get_distance_if_known()) is not None:
            return known_distance

        return min(
            (self.get_one_distance_bound(**decoder_args) for _ in range(num_trials)),
            default=len(self),
        )

    def get_one_distance_bound(self, **decoder_args: Any) -> int | float:
        """Use a randomized algorithm to compute a single upper bound on code distance."""
        raise NotImplementedError(
            "Monte Carlo distance bound calculation is not implemented for a general QuditCode"
        )

    def get_logical_ops(self, pauli: PauliXZ | None = None) -> galois.FieldArray:
        """Complete basis of nontrivial logical Pauli operators for this code.

        Logical operators are represented by a three-dimensional array `logical_ops` with dimensions
        `(2, k, 2 * n)`, where `k` and `n` are respectively the numbers of logical and physical
        qudits in this code.  The first axis is used to keep track of conjugate pairs of logical
        operators.  The last axis is "doubled" to indicate the support of physical X-type vs. Z-type
        operators.

        Specifically, each row of `logical_ops[0, :, :]` is logical X-type operator that -- due to
        the way that logical operators are constructed here -- only addresses physical qudits by
        physical X-type operators.  Each row of `logical_ops[1, :, :]` is a logical Z-type operator
        that addresses at least one physical qudit by a physical Z-type operator, and may
        additionally address physical qudits by physical X-type operators.

        For example, if `logical_ops[p, r, j] == 1` for `j < n` (`j >= n`), then the `p`-type
        logical operator for logical qudit `r` addresses physical qudit `j` with a physical X-type
        (Z-type) operator.

        If passed a pauli operator (Pauli.X or Pauli.Z), return the two-dimensional array of logical
        operators of that type.

        Logical operators are constructed using the method described in Section 4.1 of Gottesman's
        thesis (arXiv:9705052), slightly modified for qudits.
        """
        assert pauli is None or pauli in PAULIS_XZ

        # if requested, retrieve logical operators of one type only
        if pauli is not None:
            return self.get_logical_ops()[pauli]

        # memoize manually because other methods may modify the logical operators computed here
        if self._logical_ops is not None:
            return self._logical_ops

        num_qudits = self.num_qudits
        dimension = self.dimension
        identity = self.field.Identity(dimension)

        # keep track of current qudit locations
        qudit_locs = np.arange(num_qudits, dtype=int)

        # row reduce and identify pivots in the Z sector
        matrix, pivots_z = _row_reduce(self.matrix)
        pivots_z = [pivot for pivot in pivots_z if pivot < self.num_qudits]
        other_z = [qq for qq in range(self.num_qudits) if qq not in pivots_z]

        # move the Z pivots to the back
        matrix = matrix.reshape(self.num_checks * 2, self.num_qudits)
        matrix = np.hstack([matrix[:, other_z], matrix[:, pivots_z]])
        qudit_locs = np.hstack([qudit_locs[other_z], qudit_locs[pivots_z]])

        # row reduce and identify pivots in the X sector
        matrix = matrix.reshape(self.num_checks, 2 * self.num_qudits)
        sub_matrix = matrix[len(pivots_z) :, self.num_qudits :]
        sub_matrix, pivots_x = _row_reduce(self.field(sub_matrix))
        matrix[len(pivots_z) :, self.num_qudits :] = sub_matrix
        other_x = [qq for qq in range(self.num_qudits) if qq not in pivots_x]

        # move the X pivots to the back
        matrix = matrix.reshape(self.num_checks * 2, self.num_qudits)
        matrix = np.hstack([matrix[:, other_x], matrix[:, pivots_x]])
        qudit_locs = np.hstack([qudit_locs[other_x], qudit_locs[pivots_x]])

        # identify X-pivot and Z-pivot parity checks
        matrix = matrix.reshape(self.num_checks, 2 * self.num_qudits)[: len(pivots_z + pivots_x), :]
        checks_z = matrix[: len(pivots_z), :].reshape(len(pivots_z), 2, self.num_qudits)
        checks_x = matrix[len(pivots_z) :, :].reshape(len(pivots_x), 2, self.num_qudits)

        # run some sanity checks
        assert len(pivots_x) == 0 or pivots_x[-1] < num_qudits - len(pivots_z)
        assert dimension + len(pivots_z) + len(pivots_x) == num_qudits
        assert not np.any(checks_x[:, 0, :])

        # identify "sections" of columns / qudits
        section_k = slice(dimension)
        section_z = slice(dimension, dimension + len(pivots_z))
        section_x = slice(dimension + len(pivots_z), self.num_qudits)

        # construct Z-pivot logical operators
        logicals_z = self.field.Zeros((dimension, 2, num_qudits))
        logicals_z[:, 0, section_k] = identity
        logicals_z[:, 0, section_x] = -checks_x[:, 1, :dimension].T
        logicals_z[:, 1, section_z] = -(
            checks_z[:, 1, section_x] @ checks_x[:, 1, section_k] + checks_z[:, 1, section_k]
        ).T

        # construct X-pivot logical operators
        logicals_x = self.field.Zeros((dimension, 2, num_qudits))
        logicals_x[:, 1, section_k] = identity
        logicals_x[:, 1, section_z] = -checks_z[:, 0, :dimension].T

        # move qudits back to their original locations and swap X/Z sectors
        permutation = np.argsort(qudit_locs)
        logicals_x = logicals_x[:, ::-1, permutation]
        logicals_z = logicals_z[:, ::-1, permutation]

        # reshape and return
        logicals_x = logicals_x.reshape(dimension, 2 * num_qudits)
        logicals_z = logicals_z.reshape(dimension, 2 * num_qudits)
        shape = (2, self.dimension, 2 * self.num_qudits)
        self._logical_ops = self.field(np.stack([logicals_x, logicals_z]).reshape(shape))
        return self._logical_ops

    @classmethod
    def stack(cls, code_a: QuditCode, code_b: QuditCode) -> QuditCode:
        """Stack two qudit codes.

        The stacked code is obtained by having the input codes act on disjoint sets of qudits.
        Stacking two codes with parameters [n_1, k_1, d_1] and [n_2, k_2, d_2] results in a single
        code with parameters [n_1 + n_2, k_1 + k_2, min(d_1, d_2)].
        """
        if code_a.field is not code_b.field:
            raise ValueError("Cannot join codes over different fields")
        matrix_a_z = code_a.matrix.reshape(code_a.num_checks, 2, len(code_a))[:, 0, :]
        matrix_a_x = code_a.matrix.reshape(code_a.num_checks, 2, len(code_a))[:, 1, :]
        matrix_b_z = code_b.matrix.reshape(code_b.num_checks, 2, len(code_b))[:, 0, :]
        matrix_b_x = code_b.matrix.reshape(code_b.num_checks, 2, len(code_b))[:, 1, :]
        code_z = ClassicalCode.stack(ClassicalCode(matrix_a_z), ClassicalCode(matrix_b_z))
        code_x = ClassicalCode.stack(ClassicalCode(matrix_a_x), ClassicalCode(matrix_b_x))
        matrix = np.hstack([code_z.matrix, code_x.matrix])
        return QuditCode(matrix, field=code_a.field.order)


class CSSCode(QuditCode):
    """CSS qudit code, with separate X-type and Z-type parity checks.

    In order for the X-type and Z-type parity checks to be "compatible", the X-type stabilizers must
    commute with the Z-type stabilizers.  Mathematically, this requirement can be written as

    H_x @ H_z.T == 0,

    where H_x and H_z are, respectively, the parity check matrices of the classical codes that
    define the X-type and Z-type stabilizers of the CSS code.  Note that H_x witnesses Z-type errors
    and H_z witnesses X-type errors.

    The full parity check matrix of a CSSCode is
    ⌈  0 , H_x ⌉
    ⌊ H_z,  0  ⌋.
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
        skip_validation: bool = False,
    ) -> None:
        """Build a CSSCode from classical subcodes that specify X-type and Z-type parity checks."""
        self.code_x = ClassicalCode(code_x, field)
        self.code_z = ClassicalCode(code_z, field)

        if field is None and self.code_x.field is not self.code_z.field:
            raise ValueError("The sub-codes provided for this CSSCode are over different fields")
        self._field = self.code_x.field

        if not skip_validation and self.code_x != self.code_z:
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
                [np.zeros_like(self.matrix_x), self.matrix_x],
                [self.matrix_z, np.zeros_like(self.matrix_z)],
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

    def canonicalized(self) -> CSSCode:
        """The same code with its parity matrix in reduced row echelon form."""
        return CSSCode(self.code_x.canonicalized(), self.code_z.canonicalized())

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
        rank_x = self.code_x.rank
        rank_z = rank_x if self._balanced_codes else self.code_z.rank
        return rank_x + rank_z

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
        code_x = self.code_x if pauli == Pauli.X else self.code_z
        code_z = self.code_z if pauli == Pauli.X else self.code_x
        dual_code_x = ~code_x
        nontrivial_ops_x = (word for word in code_z.iter_words() if word not in dual_code_x)
        distance = min(np.count_nonzero(word) for word in nontrivial_ops_x)

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
        **decoder_args: Any,
    ) -> int | float:
        """Compute an upper bound on code distance by minimizing many individual upper bounds.

        If `pauli is not None`, consider only `pauli`-type logical operators.

        Additional arguments, if applicable, are passed to a decoder in `get_one_distance_bound`.
        """
        if (known_distance := self._get_distance_if_known(pauli)) is not None:
            return known_distance

        return min(
            (self.get_one_distance_bound(pauli, **decoder_args) for _ in range(num_trials)),
            default=self.num_qudits,
        )

    def get_one_distance_bound(
        self,
        pauli: PauliXZ | None = None,
        **decoder_args: Any,
    ) -> int | float:
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
            candidate_logical_op = decoder.decode(
                effective_check_matrix, effective_syndrome, **decoder_args
            )

            # check whether decoding was successful
            # NOTE: we can mod out by the field order because non-prime fields aren't allowed here
            actual_syndrome = effective_check_matrix @ candidate_logical_op % self.field.order
            logical_op_found = np.array_equal(actual_syndrome, effective_syndrome)

        # return the Hamming weight of the logical operator
        return int(np.count_nonzero(candidate_logical_op))

    def get_logical_ops(self, pauli: PauliXZ | None = None) -> galois.FieldArray:
        """Complete basis of nontrivial logical Pauli operators for this code.

        Logical operators are represented by a three-dimensional array `logical_ops` with dimensions
        `(2, k, 2 * n)`, where `k` and `n` are respectively the numbers of logical and physical
        qudits in this code.  The first axis is used to keep track of conjugate pairs of logical
        operators.  The last axis is "doubled" to indicate whether a physical qudit is addressed by
        a physical X-type or Z-type operator.

        Specifically, `logical_ops[0, :, :]` are logical X-type operators that address physical
        qudits by physical X-type operators, while `logical_ops[1, :, :]` are logical Z-type
        operators that address physical qudits by physical Z-type operators.

        For example, if `logical_ops[p, r, j] == 1` for `j < n` (`j >= n`), then the `p`-type
        logical operator for logical qudit `r` addresses physical qudit `j` with a physical X-type
        (Z-type) operator.  The fact that logical operators come in conjugate pairs means that
        `logical_ops(Pauli.X)[r, :] @ logical_ops(Pauli.Z)[s, :] == int(r == s)`.

        If passed a pauli operator (Pauli.X or Pauli.Z), return the two-dimensional array of logical
        operators of that type.

        Logical operators are constructed using the method described in Section 4.1 of Gottesman's
        thesis (arXiv:9705052), slightly modified for qudits and CSSCodes.
        """
        assert pauli is None or pauli in PAULIS_XZ

        # if requested, retrieve logical operators of one type only
        if pauli is not None:
            return self.get_logical_ops()[pauli]

        # memoize manually because other methods may modify the logical operators computed here
        if self._logical_ops is not None:
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
        other_x = [qq for qq in range(self.num_qudits) if qq not in pivots_x]
        checks_x = np.hstack([checks_x[:, other_x], checks_x[:, pivots_x]])
        checks_z = np.hstack([checks_z[:, other_x], checks_z[:, pivots_x]])
        qudit_locs = np.hstack([qudit_locs[other_x], qudit_locs[pivots_x]])

        # row reduce the check matrix for Z-type errors and move its pivots to the back
        checks_z, pivots_z = _row_reduce(self.field(checks_z))
        other_z = [qq for qq in range(self.num_qudits) if qq not in pivots_z]
        checks_x = np.hstack([checks_x[:, other_z], checks_x[:, pivots_z]])
        checks_z = np.hstack([checks_z[:, other_z], checks_z[:, pivots_z]])
        qudit_locs = np.hstack([qudit_locs[other_z], qudit_locs[pivots_z]])

        # run some sanity checks
        assert pivots_z[-1] < num_qudits - len(pivots_x)
        assert dimension + len(pivots_x) + len(pivots_z) == num_qudits

        # identify "sections" of columns / qudits
        section_k = slice(dimension)
        section_x = slice(dimension, dimension + len(pivots_x))
        section_z = slice(dimension + len(pivots_x), self.num_qudits)

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

        logical_ops = [
            [logicals_x, np.zeros_like(logicals_x)],
            [np.zeros_like(logicals_z), logicals_z],
        ]
        shape = (2, self.dimension, 2 * self.num_qudits)
        self._logical_ops = self.field(np.block(logical_ops).reshape(shape))
        return self._logical_ops

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

        # effective check matrix = syndromes and other logical operators
        if pauli == Pauli.X:
            code = self.code_z
            nonzero_dual_section = slice(self.num_qudits, 2 * self.num_qudits)
        else:
            code = self.code_x
            nonzero_dual_section = slice(self.num_qudits)
        all_dual_ops = self.get_logical_ops()[~pauli, :, nonzero_dual_section]
        effective_check_matrix = np.vstack([code.matrix, all_dual_ops]).view(np.ndarray)
        dual_op_index = code.num_checks + logical_index

        # enforce that the new logical operator commutes with everything except its dual
        effective_syndrome = np.zeros((code.num_checks + self.dimension), dtype=int)
        effective_syndrome[dual_op_index] = 1
        _fix_decoder_args_for_nonbinary_fields(decoder_args, self.field, bound_index=dual_op_index)

        logical_op_found = False
        while not logical_op_found:
            candidate_logical_op = decoder.decode(
                effective_check_matrix, effective_syndrome, **decoder_args
            )
            actual_syndrome = effective_check_matrix @ candidate_logical_op % self.field.order
            logical_op_found = np.array_equal(actual_syndrome, effective_syndrome)

        assert self._logical_ops is not None
        self._logical_ops.shape = (2, self.dimension, 2, self.num_qudits)
        self._logical_ops[pauli, logical_index, pauli, :] = candidate_logical_op
        self._logical_ops.shape = (2, self.dimension, 2 * self.num_qudits)

    def reduce_logical_ops(self, pauli: PauliXZ | None = None, **decoder_args: Any) -> None:
        """Reduce the weight of all logical operators."""
        assert pauli is None or pauli in PAULIS_XZ
        if pauli is None:
            self.reduce_logical_ops(Pauli.X, **decoder_args)
            self.reduce_logical_ops(Pauli.Z, **decoder_args)
        else:
            for logical_index in range(self.dimension):
                self.reduce_logical_op(pauli, logical_index, **decoder_args)

    @classmethod
    def stack(cls, code_a: QuditCode, code_b: QuditCode) -> QuditCode:
        """Stack two qudit codes.

        The stacked code is obtained by having the input codes act on disjoint sets of qudits.
        Stacking two codes with parameters [n_1, k_1, d_1] and [n_2, k_2, d_2] results in a single
        code with parameters [n_1 + n_2, k_1 + k_2, min(d_1, d_2)].

        If both input codes are CSS, the output code will likewise be a CSSCode.
        """
        if code_a.field is not code_b.field:
            raise ValueError("Cannot join codes over different fields")
        if not isinstance(code_a, CSSCode) or not isinstance(code_b, CSSCode):
            return QuditCode.stack(code_a, code_b)
        code_x = ClassicalCode.stack(code_a.code_x, code_b.code_x)
        code_z = ClassicalCode.stack(code_a.code_z, code_b.code_z)
        return CSSCode(
            code_x,
            code_z,
            field=code_a.field.order,
            promise_balanced_codes=code_a._balanced_codes and code_b._balanced_codes,
            skip_validation=True,
        )


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
