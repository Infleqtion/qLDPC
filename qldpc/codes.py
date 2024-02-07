"""Error correction code constructions

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
from collections.abc import Collection, Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Literal

import cachetools
import galois
import ldpc.mod2
import networkx as nx
import numpy as np
import numpy.typing as npt

import qldpc
from qldpc import abstract
from qldpc.objects import CayleyComplex, Node, Pauli, QuditOperator

if TYPE_CHECKING:
    from typing_extensions import Self

DEFAULT_FIELD_ORDER = abstract.DEFAULT_FIELD_ORDER

################################################################################
# template error correction code classes


# TODO(?): support sparse parity check matrices
class AbstractCode(abc.ABC):
    """Template class for error-correcting codes."""

    _field: type[galois.FieldArray]

    def __init__(
        self,
        matrix: Self | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | None = None,
    ) -> None:
        """Construct a code from a parity check matrix over a finite field.

        The base field is taken to be F_2 by default.
        """
        self._matrix: galois.FieldArray
        if isinstance(matrix, type(self)):
            self._field = matrix.field
            self._matrix = matrix.matrix
        elif isinstance(matrix, galois.FieldArray):
            self._field = type(matrix)
            self._matrix = matrix
        else:
            self._field = galois.GF(field or DEFAULT_FIELD_ORDER)
            self._matrix = self.field(np.array(matrix))

        if field is not None and field != self.field.order:
            raise ValueError(
                f"Field argument {field} is inconsistent with the given code, which is defined"
                f" over F_{self.field.order}"
            )

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field over which this code is defined."""
        return self._field

    @property
    def matrix(self) -> galois.FieldArray:
        """Parity check matrix of this code."""
        return self._matrix

    @functools.cached_property
    def graph(self) -> nx.DiGraph:
        """Tanner graph of this code."""
        return self.matrix_to_graph(self.matrix)

    @classmethod
    @abc.abstractmethod
    def matrix_to_graph(cls, matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]]) -> nx.DiGraph:
        """Convert a parity check matrix into a Tanner graph."""

    @classmethod
    @abc.abstractmethod
    def graph_to_matrix(cls, graph: nx.DiGraph) -> galois.FieldArray:
        """Convert a Tanner graph into a parity check matrix."""


class ClassicalCode(AbstractCode):
    """Classical linear error-correcting code over a finite field F_q.

    A classical binary code C = {x} is a set of vectors x (with entries in F_q) called code words.
    We consider only linear codes, for which any linear combination of code words is also code word.

    Operationally, we define a classical code by a parity check matrix H with dimensions
    (num_checks, num_bits).  Each row of H represents a linear constraint (a "check") that code
    words must satisfy.  A vector x is a code word iff H @ x = 0.
    """

    def __contains__(self, word: npt.NDArray[np.int_] | Sequence[int]) -> bool:
        return not np.any(self.matrix @ self.field(word))

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
        field = graph.order if hasattr(graph, "order") else DEFAULT_FIELD_ORDER
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

    def get_random_word(self) -> galois.FieldArray:
        """Random code word: a sum all generators with random field coefficients."""
        return self.field.Random(self.generator.shape[0]) @ self.generator

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
        """Tensor product C_a ⊗ C_b of two codes C_a and C_b.

        Let G_a and G_b respectively denote the generators C_a and C_b.
        Definition: C_a ⊗ C_b is the code whose generators are G_a ⊗ G_b.

        Observation: G_a ⊗ G_b is the check matrix of ~(C_a ⊗ C_b).
        We therefore construct ~(C_a ⊗ C_b) and return its dual ~~(C_a ⊗ C_b) = C_a ⊗ C_b.
        """
        if code_a.field is not code_b.field:
            raise ValueError("Cannot take tensor product of codes over different fields")
        gen_a: npt.NDArray[np.int_] = code_a.generator
        gen_b: npt.NDArray[np.int_] = code_b.generator
        return ~ClassicalCode(np.kron(gen_a, gen_b))

    @property
    def num_checks(self) -> int:
        """Number of check bits in this code."""
        return self._matrix.shape[0]

    @property
    def num_bits(self) -> int:
        """Number of data bits in this code."""
        return self._matrix.shape[1]

    @functools.cached_property
    def rank(self) -> int:
        """Rank of this code's parity check matrix.

        Equivalently, the number of linearly independent parity checks in this code.
        """
        if self.field.order == 2:
            return ldpc.mod2.rank(self._matrix)
        return np.linalg.matrix_rank(self._matrix)

    @property
    def dimension(self) -> int:
        """The number of logical bits encoded by this code."""
        return self.num_bits - self.rank

    @functools.cache
    def get_distance(self) -> int:
        """The distance of this code, or equivalently the minimal weight of a nonzero code word."""
        words = self.words().view(np.ndarray)
        return np.min(np.count_nonzero(words[1:], axis=1))

    def get_code_params(self) -> tuple[int, int, int]:
        """Compute the parameters of this code: [n,k,d].

        Here:
        - n is the number of data bits
        - k is the number of encoded ("logical") bits
        - d is the code distance
        """
        return self.num_bits, self.dimension, self.get_distance()

    @classmethod
    def random(cls, bits: int, checks: int, field: int | None = None) -> ClassicalCode:
        """Construct a random classical code with the given number of bits and nontrivial checks."""
        code_field = galois.GF(field or DEFAULT_FIELD_ORDER)
        rows, cols = checks, bits
        matrix = code_field.Random((rows, cols))
        for row in range(matrix.shape[0]):
            if not matrix[row, :].any():
                matrix[row, np.random.randint(cols)] = code_field.Random(low=1)  # pragma: no cover
        for col in range(matrix.shape[1]):
            if not matrix[:, col].any():
                matrix[np.random.randint(rows), col] = code_field.Random(low=1)  # pragma: no cover
        return ClassicalCode(matrix)

    @classmethod
    def repetition(cls, num_bits: int, field: int | None = None) -> ClassicalCode:
        """Construct a repetition code on the given number of bits."""
        code_field = galois.GF(field or DEFAULT_FIELD_ORDER)
        matrix = code_field.Zeros((num_bits - 1, num_bits))
        for row in range(num_bits - 1):
            matrix[row, row] = 1
            matrix[row, row + 1] = -code_field(1)
        return ClassicalCode(matrix)

    @classmethod
    def ring(cls, num_bits: int, field: int | None = None) -> ClassicalCode:
        """Construct a repetition code with periodic boundary conditions."""
        code_field = galois.GF(field or DEFAULT_FIELD_ORDER)
        matrix = code_field.Zeros((num_bits, num_bits))
        for row in range(num_bits):
            matrix[row, row] = 1
            matrix[row, (row + 1) % num_bits] = -code_field(1)
        return ClassicalCode(matrix)

    @classmethod
    def hamming(cls, rank: int, field: int | None = None) -> ClassicalCode:
        """Construct a hamming code of a given rank."""
        field = field or DEFAULT_FIELD_ORDER
        if field == 2:
            # parity check matrix: columns = all nonzero bitstrings
            bitstrings = list(itertools.product([0, 1], repeat=rank))
            return ClassicalCode(np.array(bitstrings[1:]).T)

        # More generally, columns = maximal set of nonzero, linearly independent strings.
        # This is achieved by collecting together all strings whose first nonzero element is a 1.
        strings = [
            (0,) * top_row + (1,) + rest
            for top_row in range(rank - 1, -1, -1)
            for rest in itertools.product(range(field), repeat=rank - top_row - 1)
        ]
        return ClassicalCode(np.array(strings).T, field=field)

    # TODO: add more codes, particularly from code families that are useful for good quantum codes
    # see https://mhostetter.github.io/galois/latest/api/#forward-error-correction


# TODO:
# - add is_CSS method to figure out whether this is a CSS Code
#   - see https://quantumcomputing.stackexchange.com/questions/15432/
#   - also compute and store sub-codes, if CSS
#   - also add QuditCode.to_CSS() -> CSSCode
class QuditCode(AbstractCode):
    """Quantum stabilizer code for Galois qudits, with dimension q = p^m for prime p and integer m.

    The parity check matrix of a QuditCode has dimensions (num_checks, 2 * num_qudits), and can be
    written as a block matrix in the form H = [H_x|H_z].  Each block has num_qudits columns.

    The entries H_x[c, d] = r_x and H_z[c, d] = r_z iff check c addresses qudit d with the operator
    X(r_x) * Z(r_z), where r_x, r_z range over the base field, and X(r), Z(r) are generalized Pauli
    operators.  Specifically:
    - X(r) = sum_{j=0}^{q-1} |j+r><j| is a shift operator, and
    - Z(r) = sum_{j=0}^{q-1} w^{j r} |j><j| is a phase operator, with w = exp(2 pi i / q).

    Warning: here j, r, s, etc. not integers, but elements of the Galois field GF(q), which has
    different rules for addition and multiplication when q is not a prime number.

    Helpful lecture by Gottesman: https://www.youtube.com/watch?v=JWg4zrNAF-g
    """

    @property
    def num_checks(self) -> int:
        """Number of parity checks (stabilizers) in this code."""
        return self.matrix.shape[0]

    @property
    def num_qudits(self) -> int:
        """Number of data qudits in this code."""
        return self.matrix.shape[1] // 2

    @property
    def num_qubits(self) -> int:
        """Number of data qubits in this code."""
        self._assert_qubit_code()
        return self.num_qudits

    def _assert_qubit_code(self) -> None:
        if self.field.order != 2:
            raise ValueError("Attempted to call a qubit-only method with a non-qubit code")

    @classmethod
    def matrix_to_graph(cls, matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]]) -> nx.DiGraph:
        """Convert a parity check matrix into a Tanner graph."""
        graph = nx.DiGraph()
        matrix = np.reshape(matrix, (len(matrix), 2, -1))
        for row, col_xz, col in zip(*np.nonzero(matrix)):
            node_check = Node(index=int(row), is_data=False)
            node_qudit = Node(index=int(col), is_data=True)
            graph.add_edge(node_check, node_qudit)

            qudit_op = graph[node_check][node_qudit].get(QuditOperator, QuditOperator())
            vals_xz = list(qudit_op.value)
            vals_xz[col_xz] += int(matrix[row, col_xz, col])
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
                val_x = matrix[check, Pauli.X.index, qudit]
                val_z = matrix[check, Pauli.Z.index, qudit]
                vals_xz = (val_x, val_z)
                if self.field.order == 2:
                    ops.append(str(Pauli(vals_xz)))
                else:
                    ops.append(str(QuditOperator(vals_xz)))
            stabilizers.append(" ".join(ops))
        return stabilizers

    @classmethod
    def from_stabilizers(cls, stabilizers: Iterable[str], field: int | None = None) -> QuditCode:
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

        return QuditCode(matrix.reshape(num_checks, 2 * num_qudits), field)

    # TODO: generalize to any local Clifford deformation
    #       see https://arxiv.org/pdf/quant-ph/0408190.pdf
    @classmethod
    def conjugate(
        cls, matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]], qudits: slice | Sequence[int]
    ) -> npt.NDArray[np.int_]:
        """Apply local Fourier transforms to the given qudits.

        This is equivalent to swapping X-type and Z-type operators."""
        num_checks = len(matrix)
        matrix = np.reshape(matrix, (num_checks, 2, -1))
        matrix[:, :, qudits] = np.roll(matrix[:, :, qudits], 1, axis=1)
        return matrix.reshape(num_checks, -1)


class CSSCode(QuditCode):
    """CSS qudit code, with separate X-type and Z-type parity checks.

    In order for the X-type and Z-type parity checks to be "compatible", the X-type stabilizers must
    commute with the Z-type stabilizers.  Mathematically, this requirement can be written as

    H_x @ H_z.T == 0,

    where H_x and H_z are, respectively, the parity check matrices of the classical codes that
    define the X-type and Z-type stabilizers of the CSS code.  Note that H_x witnesses Z-type errors
    and H_z witnesses X-type errors.

    The full parity check matrix of a CSSCode is
    ⌈ H_z,  0  ⌉
    ⌊  0 , H_x ⌋.
    """

    code_x: ClassicalCode  # X-type parity checks, measuring Z-type errors
    code_z: ClassicalCode  # Z-type parity checks, measuring X-type errors
    _field_order: int  # The order of the field over which the CSS code is defined

    _conjugate: slice | Sequence[int]
    _codes_equal: bool
    _logical_ops: galois.FieldArray | None = None

    def __init__(
        self,
        code_x: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_z: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | None = None,
        *,
        conjugate: slice | Sequence[int] | None = (),
        skip_validation: bool = False,
    ) -> None:
        """Construct a CSS code from X-type and Z-type parity checks.

        Allow specifying local Fourier transformations on the qudits specified by `conjugate`.
        """
        self.code_x = ClassicalCode(code_x, field)
        self.code_z = ClassicalCode(code_z, field)
        if field is None and self.code_x.field is not self.code_z.field:
            raise ValueError("The sub-codes provided for this CSSCode are over different fields")
        self._field = self.code_x.field

        if not skip_validation and not self.is_valid:
            raise ValueError("The sub-codes provided for this CSSCode are incompatible")

        self._conjugate = conjugate or ()
        self._codes_equal = self.code_x == self.code_z

    @functools.cached_property
    def is_valid(self) -> bool:
        """Is this a valid CSS code?"""
        return self.code_x.num_bits == self.code_z.num_bits and not np.any(
            self.code_x.matrix @ self.code_z.matrix.T
        )

    @functools.cached_property
    def matrix(self) -> galois.FieldArray:
        """Overall parity check matrix."""
        matrix = np.block(
            [
                [self.code_z.matrix, np.zeros_like(self.code_z.matrix)],
                [np.zeros_like(self.code_x.matrix), self.code_x.matrix],
            ]
        )
        return self.field(self.conjugate(matrix, self._conjugate))

    @property
    def num_checks(self) -> int:
        """Number of parity checks in this code."""
        return self.code_x.matrix.shape[0] + self.code_z.matrix.shape[0]

    @property
    def num_qudits(self) -> int:
        """Number of data qubits in this code."""
        return self.code_x.matrix.shape[1]

    @property
    def dimension(self) -> int:
        """Number of logical qudits encoded by this code."""
        return self.code_x.dimension + self.code_z.dimension - self.num_qudits

    def get_code_params(
        self, *, lower: bool = False, upper: int | None = None, **decoder_args: object
    ) -> tuple[int, int, int]:
        """Compute the parameters of this code: [[n,k,d]].

        Here:
        - n is the number of data qudits
        - k is the number of encoded ("logical") qudits
        - d is the code distance

        Keyword arguments are passed to the calculation of code distance.
        """
        distance = self.get_distance(pauli=None, lower=lower, upper=upper, **decoder_args)
        return self.num_qudits, self.dimension, distance

    def get_distance(
        self,
        pauli: Literal[Pauli.X, Pauli.Z] | None = None,
        *,
        lower: bool = False,
        upper: int | None = None,
        **decoder_args: object,
    ) -> int:
        """Distance of the this code: minimum weight of a nontrivial logical operator.

        If provided a Pauli as an argument, compute the minimim weight of an nontrivial logical
        operator of the corresponding type.  Otherwise, minimize over Pauli.X and Pauli.Z.

        If `lower is True`, compute a lower bound: for the X-distance, compute the distance of the
        classical Z-type subcode that corrects X-type errors.  Vice versa with the Z-distance.

        If `upper is not None`, compute an upper bound using a randomized algorithm described in
        arXiv:2308.07915, minimizing over `upper` random trials.  For a detailed explanation, see
        `CSSCode.get_distance_upper_bound` and `CSSCode.get_one_distance_upper_bound`.

        If `lower is False` and `upper is None`, compute an exact code distance with integer linear
        programming.  Warning: this is an NP-complete problem and takes exponential time to execute.

        All remaining keyword arguments are passed to a decoder, if applicable.
        """
        if lower and upper:
            raise ValueError(
                "Must choose between computing lower and upper bounds on code distance"
            )
        assert pauli == Pauli.X or pauli == Pauli.Z or pauli is None
        pauli = pauli if not self._codes_equal else Pauli.X

        if pauli is None:
            return min(
                self.get_distance(Pauli.X, lower=lower, upper=upper, **decoder_args),
                self.get_distance(Pauli.Z, lower=lower, upper=upper, **decoder_args),
            )

        if lower:
            return self.get_distance_lower_bound(pauli)

        if upper is not None:
            return self.get_distance_upper_bound(pauli, num_trials=upper, **decoder_args)

        return self.get_distance_exact(pauli, **decoder_args)

    def get_distance_lower_bound(self, pauli: Literal[Pauli.X, Pauli.Z]) -> int:
        """Lower bound to the X-distance or Z-distance of this code."""
        assert pauli == Pauli.X or pauli == Pauli.Z
        pauli = pauli if not self._codes_equal else Pauli.X
        return self.code_z.get_distance() if pauli == Pauli.X else self.code_x.get_distance()

    def get_distance_upper_bound(
        self,
        pauli: Literal[Pauli.X, Pauli.Z],
        num_trials: int,
        **decoder_args: object,
    ) -> int:
        """Upper bound to the X-distance or Z-distance of this code, minimized over many trials.

        All keyword arguments are passed to `CSSCode.get_one_distance_upper_bound`.
        """
        assert pauli == Pauli.X or pauli == Pauli.Z
        return min(
            self.get_one_distance_upper_bound(pauli, **decoder_args) for _ in range(num_trials)
        )

    def get_one_distance_upper_bound(
        self,
        pauli: Literal[Pauli.X, Pauli.Z],
        **decoder_args: object,
    ) -> int:
        """Single upper bound to the X-distance or Z-distance of this code.

        This method uses a randomized algorithm described in arXiv:2308.07915 (and also below).

        Args:
            pauli: Pauli operator choosing whether to compute an X-distance or Z-distance bound.
            decoder_args: Keyword arguments are passed to a decoder in `decode`.
        Returns:
            An upper bound on the X-distance or Z-distance of this code.

        For ease of language, we henceforth assume that we are computing an X-distance.

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
        find a solution, try again with a new initial random operator Z(w_z).  If the decoder
        succeeds in finding a solution w_x, this solution corresponds to a logical X-type operator
        X(w_x) -- and presumably one of low Hamming weight, since decoders try to find low-weight
        solutions.  Return the Hamming weight |w_x|.
        """
        assert pauli == Pauli.X or pauli == Pauli.Z
        if self.field.order != 2:
            raise ValueError(
                "Distance upper bound calculation not implemented for fields of order > 2"
            )

        # define code_z and pauli_z as if we are computing X-distance
        code_z = self.code_z if pauli == Pauli.X else self.code_x
        pauli_z: Literal[Pauli.Z, Pauli.X] = Pauli.Z if pauli == Pauli.X else Pauli.X

        # construct the effective syndrome
        effective_syndrome = np.zeros(code_z.num_checks + 1, dtype=int)
        effective_syndrome[-1] = 1

        logical_op_found = False
        while not logical_op_found:
            # support of pauli string with a trivial syndrome
            word = self.get_random_logical_op(pauli_z)

            # support of a candidate pauli-type logical operator
            effective_check_matrix = np.vstack([code_z.matrix, word]).view(np.ndarray)
            candidate_logical_op = qldpc.decoder.decode(
                effective_check_matrix, effective_syndrome, exact=False, **decoder_args
            )

            # check whether the decoding was successful
            actual_syndrome = effective_check_matrix @ candidate_logical_op % 2
            logical_op_found = np.array_equal(actual_syndrome, effective_syndrome)

        # return the Hamming weight of the logical operator
        return candidate_logical_op.sum()

    @cachetools.cached(cache={}, key=lambda self, pauli, **decoder_args: (self, pauli))
    def get_distance_exact(self, pauli: Literal[Pauli.X, Pauli.Z]) -> int:
        """Exact X-distance or Z-distance of this code."""
        assert pauli == Pauli.X or pauli == Pauli.Z
        pauli = pauli if not self._codes_equal else Pauli.X
        if self.field.degree > 1:
            # The base field is not prime, so we can't use the integer linear program method due to
            # different rules for addition and multiplication.  We therefore compute distance with a
            # brute-force search over all code words.
            code_x = self.code_x if pauli == Pauli.X else self.code_z
            code_z = self.code_z if pauli == Pauli.X else self.code_x
            dual_code_x = ~code_x
            return min(np.count_nonzero(word) for word in code_z.words() if word not in dual_code_x)

        # TODO: is this wrong???

        # minimize the weight of logical X-type or Z-type operators
        for logical_qubit_index in range(self.dimension):
            self.minimize_logical_op(pauli, logical_qubit_index)

        # return the minimum weight of logical X-type or Z-type operators
        return np.count_nonzero(self.get_logical_ops()[pauli.index].view(np.ndarray), axis=-1).min()

    def get_logical_ops(self) -> galois.FieldArray:
        """Complete basis of nontrivial X-type and Z-type logical operators for this code.

        Logical operators are represented by a three-dimensional array `logical_ops` with dimensions
        (2, k, n), where k and n are respectively the numbers of logical and physical qubits in this
        code.  The bitstring `logical_ops[0, 4, :]`, for example, indicates the support (i.e., the
        physical qubits addressed nontrivially) by the logical Pauli-X operator on logical qubit 4.

        In the case of qudits with dimension > 2, the "Pauli-X" and "Pauli-Z" operators constructed
        by this method are the unit shift and phase operators that generate all logical X-type and
        Z-type qudit operators.

        Logical operators are constructed using the method described in Section 4.1 of Gottesman's
        thesis (arXiv:9705052), slightly modified and generalized for qudits.
        """
        # memoize manually because other methods may modify the logical operators computed here
        if self._logical_ops is not None:
            return self._logical_ops

        num_qudits = self.num_qudits
        dimension = self.dimension
        identity = self.field.Identity(dimension)

        def row_reduce(
            matrix: npt.NDArray[np.int_],
        ) -> tuple[npt.NDArray[np.int_], Sequence[int], Sequence[int]]:
            """Perform Gaussian elimination on the matrix.

            Returns:
                matrix_RRE: the reduced row echelon form of the matrix.
                pivot: the "pivot" columns of the reduced matrix.
                other: the remaining columns of the reduced matrix.

            In reduced row echelon form, the first nonzero entry of each row is a 1, and these 1s
            occur at a unique columns for each row; these columns are the "pivots" of matrix_RRE.
            """
            # row-reduce the matrix and identify its pivots
            matrix_RRE = self.field(matrix).row_reduce()
            pivots = (matrix_RRE != 0).argmax(axis=1)

            # remove trailing zero pivots, which correspond to trivial (all-zero) rows
            if pivots.size > 1 and pivots[-1] == 0:
                pivots = np.concatenate([[pivots[0]], pivots[1:][pivots[1:] != 0]])

            # identify remaining columns and return
            other = [qq for qq in range(matrix.shape[1]) if qq not in pivots]
            return matrix_RRE, pivots, other

        # identify check matrices for X/Z-type errors, and the current qubit locations
        checks_x: npt.NDArray[np.int_] = self.code_z.matrix
        checks_z: npt.NDArray[np.int_] = self.code_x.matrix
        qubit_locs = np.arange(num_qudits, dtype=int)

        # row reduce the check matrix for X-type errors and move its pivots to the back
        checks_x, pivot_x, other_x = row_reduce(checks_x)
        checks_x = np.hstack([checks_x[:, other_x], checks_x[:, pivot_x]])
        checks_z = np.hstack([checks_z[:, other_x], checks_z[:, pivot_x]])
        qubit_locs = np.hstack([qubit_locs[other_x], qubit_locs[pivot_x]])

        # row reduce the check matrix for Z-type errors and move its pivots to the back
        checks_z, pivot_z, other_z = row_reduce(checks_z)
        checks_x = np.hstack([checks_x[:, other_z], checks_x[:, pivot_z]])
        checks_z = np.hstack([checks_z[:, other_z], checks_z[:, pivot_z]])
        qubit_locs = np.hstack([qubit_locs[other_z], qubit_locs[pivot_z]])

        # run some sanity checks
        assert pivot_z[-1] < num_qudits - len(pivot_x)
        assert dimension + len(pivot_x) + len(pivot_z) == num_qudits

        # get the support of the check matrices on non-pivot qudits
        non_pivot_x = checks_x[: len(pivot_x), :dimension]
        non_pivot_z = checks_z[: len(pivot_z), :dimension]

        # construct logical X operators
        logicals_x = self.field.Zeros((dimension, num_qudits))
        logicals_x[:, dimension : dimension + len(pivot_x)] = -non_pivot_x.T
        logicals_x[:dimension, :dimension] = identity

        # construct logical Z operators
        logicals_z = self.field.Zeros((dimension, num_qudits))
        logicals_z[:, -len(pivot_z) :] = -non_pivot_z.T
        logicals_z[:dimension, :dimension] = identity

        # move qudits back to their original locations
        permutation = np.argsort(qubit_locs)
        logicals_x = logicals_x[:, permutation]
        logicals_z = logicals_z[:, permutation]

        self._logical_ops = self.field(np.stack([logicals_x, logicals_z]))
        return self._logical_ops

    def get_random_logical_op(
        self, pauli: Literal[Pauli.X, Pauli.Z], ensure_nontrivial: bool = False
    ) -> galois.FieldArray:
        """Return a random logical operator of a given type.

        A random logical operator may be trivial, which is to say that it may be equal to the
        identity modulo stabilizers.  If `ensure_nontrivial is True`, ensure that the logical
        operator we return is nontrivial.
        """
        assert pauli == Pauli.X or pauli == Pauli.Z
        if ensure_nontrivial:
            random_logical_qudit_index = np.random.randint(self.dimension)
            return self.get_logical_ops()[pauli.index, random_logical_qudit_index]
        return (self.code_z if pauli == Pauli.X else self.code_x).get_random_word()

    def minimize_logical_op(
        self, pauli: Literal[Pauli.X, Pauli.Z], logical_qubit_index: int
    ) -> None:
        """Minimize the weight of a logical operator.

        A minimum-weight logical operator is found by enforcing that it has a trivial syndrome, and
        that it commutes with all logical operators except its dual.  This is essentially the same
        optimization as in CSSCode.get_one_distance_upper_bound, but solved exactly with integer
        linear programming.
        """
        assert pauli == Pauli.X or pauli == Pauli.Z
        assert 0 <= logical_qubit_index < self.dimension
        if self.field.degree > 1:
            raise ValueError("Method only supported for prime number fields")

        # effective check matrix = syndromes and other logical operators
        code = self.code_z if pauli == Pauli.X else self.code_x
        dual_ops = self.get_logical_ops()[(~pauli).index]
        effective_check_matrix = np.vstack([code.matrix, dual_ops]).view(np.ndarray)

        # enforce that the new logical operator commutes with everything except its dual
        effective_syndrome = np.zeros((code.num_checks + self.dimension), dtype=int)
        effective_syndrome[code.num_checks + logical_qubit_index] = 1

        logical_op = qldpc.decoder.decode(
            effective_check_matrix,
            effective_syndrome,
            exact=True,
            modulus=self.field.order,
        )
        assert self._logical_ops is not None
        self._logical_ops[pauli.index, logical_qubit_index] = logical_op


################################################################################
# bicycle and quasi-cyclic codes


# TODO: add special/simpler cases of code distance calculations and bounds, if and where available


class GBCode(CSSCode):
    """Generalized bicycle (GB) code.

    A GBCode code is built out of two square matrices A and B, which are combined as
    - matrix_x = [A, B.T], and
    - matrix_z = [B, A.T],
    to form the parity check matrices of a CSSCode.  As long as A and B.T commute, the parity check
    matrices matrix_x and matrix_z satisfy the requirements of a CSSCode by construction.

    References:
    - https://arxiv.org/abs/2012.04068
    """

    def __init__(
        self,
        matrix_a: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        matrix_b: npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        field: int | None = None,
        *,
        conjugate: slice | Sequence[int] = (),
    ) -> None:
        """Construct a generalized bicycle code."""
        if matrix_b is None:
            matrix_b = matrix_a  # pragma: no cover
        matrix_a = np.array(matrix_a)
        matrix_b = np.array(matrix_b)
        if not np.array_equal(matrix_a @ matrix_b.T, matrix_b.T @ matrix_a):
            raise ValueError("The matrices provided for this GBCode are incompatible")
        matrix_x = np.block([matrix_a, matrix_b.T])
        matrix_z = np.block([matrix_b, matrix_a.T])
        CSSCode.__init__(self, matrix_x, matrix_z, field, conjugate=conjugate, skip_validation=True)


class QCCode(GBCode):
    """Quasi-cyclic (QC) code.

    Inspired by arXiv:2308.07915.

    A quasi-cyclic code is a CSS code with subcode parity check matrices
    - matrix_x = [A, B.T], and
    - matrix_z = [B, A.T],
    where A and B are block matrices identified with elements of a multivariate polynomial ring.
    Specifically, we can expand (say) A = sum_{i,j} A_{ij} x_i^j, where A_{ij} are coefficients
    and each x_i is the generator of a cyclic group of order R_i.

    We (tentatively) restrict the coefficients A_{ij} to be in {0, 1}.

    A quasi-cyclic code is defined by...
    [1] sequence (R_0, R_1, ...) of cyclic group orders (one per variable, x_i), and
    [2] a list of nonzero terms in A and B, with the term x_i^j identified by the tuple (i, j).
    The polynomial A = x + y^3 + z^2, for example, is identified by [(0, 1), (1, 3), (2, 2)].
    """

    def __init__(
        self,
        dims: Sequence[int],
        terms_a: Collection[tuple[Hashable, int]],
        terms_b: Collection[tuple[Hashable, int]] | None = None,
        field: int | None = None,
        *,
        conjugate: slice | Sequence[int] = (),
    ) -> None:
        """Construct a quasi-cyclic code."""
        if field and field != 2:
            raise ValueError("Non-boolean (field > 2) quasi-cyclic codes are not supported")

        if terms_b is None:
            terms_b = terms_a  # pragma: no cover

        # identify the symbols used to denote cyclic group generators
        symbols = tuple({symbol for symbol, _ in list(terms_a) + list(terms_b)})
        if len(symbols) != len(dims):
            raise ValueError(
                f"Number of cyclic group orders, {dims}, does not match the number of generator"
                f" symbols, {symbols}"
            )

        # identify the base cyclic groups, their product, and the generators
        groups = [abstract.CyclicGroup(dim) for dim in dims]
        group = abstract.Group.product(*groups)
        generators = group.generators

        # build defining matrices of a generalized bicycle code
        members_a = [generators[symbols.index(ss)] ** pp for ss, pp in terms_a]
        members_b = [generators[symbols.index(ss)] ** pp for ss, pp in terms_b]
        matrix_a = abstract.Element(group, *members_a).lift()
        matrix_b = abstract.Element(group, *members_b).lift()
        GBCode.__init__(self, matrix_a, matrix_b, field, conjugate=conjugate)


################################################################################
# hypergraph and lifted product codes


class HGPCode(CSSCode):
    """Hypergraph product (HGP) code.

    A hypergraph product code AB is constructed from two classical codes, A and B.

    Consider the following:
    - Code A has 3 data and 2 check bits.
    - Code B has 4 data and 3 check bits.
    We represent data bits/qudits by circles (○) and check bits/qudits by squares (□).

    Denode the Tanner graph of code C by G_C.  The nodes of G_AB can be arranged into a matrix.  The
    rows of this matrix are labeled by nodes of G_A, and columns by nodes of G_B.  The matrix of
    nodes in G_AB can thus be organized into four sectors:

    ――――――――――――――――――――――――――――――――――
      | ○ ○ ○ ○ | □ □ □ ← nodes of G_B
    ――+―――――――――+――――――
    ○ | ○ ○ ○ ○ | □ □ □
    ○ | ○ ○ ○ ○ | □ □ □
    ○ | ○ ○ ○ ○ | □ □ □
    ――+―――――――――+――――――
    □ | □ □ □ □ | ○ ○ ○
    □ | □ □ □ □ | ○ ○ ○
    ↑ nodes of G_A
    ――――――――――――――――――――――――――――――――――

    We identify each sector by two bits.
    In the example above:
    - sector (0, 0) has 3×4=12 data qudits
    - sector (0, 1) has 3×3=9 check qudits
    - sector (1, 0) has 2×4=8 check qudits
    - sector (1, 1) has 2×3=6 data qudits

    Edges in G_AB are inherited across rows/columns from G_A and G_B.  For example, if rows r_1 and
    r_2 share an edge in G_A, then the same is true in every column of G_AB.

    By default, the check qubits in sectors (0, 1) of G_AB measure Z-type operators.  Likewise with
    sector (1, 0) and X-type operators.  If a HGP is constructed with `conjugate==True`, then the
    types of operators addressing the nodes in sector (1, 1) are switched.

    This class contains two equivalent constructions of an HGPCode:
    - A construction based on Tanner graphs (as discussed above).
    - A construction based on check matrices, taken from arXiv:2202.01702.
    The latter construction is less intuitive, but more efficient.

    References:
    - https://errorcorrectionzoo.org/c/hypergraph_product
    - https://arxiv.org/abs/2202.01702
    - https://www.youtube.com/watch?v=iehMcUr2saM
    - https://arxiv.org/abs/0903.0566
    - https://arxiv.org/abs/1202.0928
    """

    sector_size: npt.NDArray[np.int_]

    def __init__(
        self,
        code_a: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_b: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        field: int | None = None,
        *,
        conjugate: bool = False,
    ) -> None:
        """Hypergraph product of two classical codes, as in arXiv:2202.01702.

        The parity check matrices of the hypergraph product code are:

        matrix_x = [H1 ⊗ In2, -Im1 ⊗ H2.T]
        matrix_z = [In1 ⊗ H2,  H1.T ⊗ Im2]

        Here (H1, H2) == (matrix_a, matrix_b), and I[m/n][1/2] are identity matrices,
        with (m1, n1) = H1.shape and (m2, n2) = H2.shape.
        """
        if code_b is None:
            code_b = code_a
        code_a = ClassicalCode(code_a, field)
        code_b = ClassicalCode(code_b, field)
        field = code_a.field.order

        # identify the number of qudits in each sector
        self.sector_size = np.outer(
            [code_a.num_bits, code_a.num_checks],
            [code_b.num_bits, code_b.num_checks],
        )
        qudits_to_conjugate = slice(self.sector_size[0, 0], None) if conjugate else None

        # construct the nontrivial blocks of the parity check matrices
        matrix_a = code_a.matrix
        matrix_b = code_b.matrix
        mat_H1_In2 = np.kron(matrix_a, np.eye(matrix_b.shape[1], dtype=int))
        mat_In1_H2 = np.kron(np.eye(matrix_a.shape[1], dtype=int), matrix_b)
        mat_H1_Im2_T = np.kron(matrix_a.T, np.eye(matrix_b.shape[0], dtype=int))
        mat_Im1_H2_T = np.kron(np.eye(matrix_a.shape[0], dtype=int), matrix_b.T)

        # construct the parity check matrices
        matrix_x = np.block([mat_H1_In2, -mat_Im1_H2_T])
        matrix_z = np.block([mat_In1_H2, mat_H1_Im2_T])
        CSSCode.__init__(
            self, matrix_x, matrix_z, field, conjugate=qudits_to_conjugate, skip_validation=True
        )

    @classmethod
    def get_graph_product(
        cls, graph_a: nx.DiGraph, graph_b: nx.DiGraph, *, conjugate: bool = False
    ) -> nx.DiGraph:
        """Hypergraph product of two Tanner graphs."""

        # start with a cartesian products of the input graphs
        graph_product = nx.cartesian_product(graph_a, graph_b)

        # fix edge orientation, and tag each edge with a QuditOperator
        graph = nx.DiGraph()
        for node_fst, node_snd, data in graph_product.edges(data=True):
            # determine which node is a check node vs. a qudit node
            if node_fst[0].is_data == node_fst[1].is_data:
                # the first node is in the (0, 0) or (1, 1) sector --> a data node
                node_qudit, node_check = node_fst, node_snd
            else:
                # the first node is in the (0, 1) or (1, 0) sector --> a check node
                node_check, node_qudit = node_fst, node_snd
            graph.add_edge(node_check, node_qudit)

            # by default, this edge is Z-type iff the check qudit is in the (0, 1) sector
            op = QuditOperator((0, data.get("val", 1)))
            if node_check[0].is_data:
                # make this a X-type operator
                op = ~op

            # special treatment of qudits in the (1, 1) sector
            if not node_qudit[0].is_data:
                # account for the minus sign in the X-type subcode
                if not node_check[0].is_data:
                    op = -op
                # flip X <--> Z operators for the conjugated code
                if conjugate:
                    op = ~op

            graph[node_check][node_qudit][QuditOperator] = op

        # relabel nodes, from (node_a, node_b) --> node_combined
        node_map = HGPCode.get_product_node_map(graph_a.nodes, graph_b.nodes)
        graph = nx.relabel_nodes(graph, node_map)

        # remember order of the field, and use Pauli operators if appropriate
        if hasattr(graph_a, "order"):
            graph.order = graph_a.order
            if graph.order == 2:
                for _, __, data in graph.edges(data=True):
                    data[Pauli] = Pauli(data[QuditOperator].value)
                    del data[QuditOperator]

        return graph

    @classmethod
    def get_product_node_map(
        cls, nodes_a: Collection[Node], nodes_b: Collection[Node]
    ) -> dict[tuple[Node, Node], Node]:
        """Map (dictionary) that re-labels nodes in the hypergraph product of two codes."""
        index_qudit = 0
        index_check = 0
        node_map = {}
        for node_a, node_b in itertools.product(sorted(nodes_a), sorted(nodes_b)):
            if node_a.is_data == node_b.is_data:
                # this is a data qudit in sector (0, 0) or (1, 1)
                node = Node(index=index_qudit, is_data=True)
                index_qudit += 1
            else:
                # this is a check qudit in sector (0, 1) or (1, 0)
                node = Node(index=index_check, is_data=False)
                index_check += 1
            node_map[node_a, node_b] = node
        return node_map


class LPCode(CSSCode):
    """Lifted product (LP) code.

    A lifted product code is essentially the same as a hypergraph product code, except that the
    parity check matrices are "protographs", or matrices whose entries are members of a group
    algebra over the field {0, 1}.  Each of these entries can be "lifted" to a representation as
    orthogonal matrices over a finite field, in which case the protograph is interpreted as a block
    matrix; this is called "lifting" the protograph.

    Notes:
    - A lifted product code with protographs of size 1×1 is a generalized bicycle code.

    References:
    - https://errorcorrectionzoo.org/c/lifted_product
    - https://arxiv.org/abs/2202.01702
    - https://arxiv.org/abs/2012.04068
    """

    def __init__(
        self,
        protograph_a: abstract.Protograph | npt.NDArray[np.object_] | Sequence[Sequence[object]],
        protograph_b: abstract.Protograph
        | npt.NDArray[np.object_]
        | Sequence[Sequence[object]]
        | None = None,
        *,
        conjugate: bool = False,
    ) -> None:
        """Same hypergraph product as in the HGPCode, but with protographs.

        There is one crucial subtlety when computing the hypergraph product of protographs.  When
        taking the transpose of a protograph, P --> P.T, we also need to transpose the individual
        (algebra-valued) entries of the protograph.  That is,

        P = ⌈ a, b ⌉  ==>  P.T = ⌈ a.T, c.T ⌉
            ⌊ c, d ⌋             ⌊ b.T, d.T ⌋.

        If we simply take the hypergraph product of two protograph matrices directly, numpy will not
        know to take the transpose of matrix entries when taking the transpose of a matrix.  For
        this reason, we need to take the transpose of a protograph "manually" when using it for the
        hypergraph product.
        """
        if protograph_b is None:
            protograph_b = protograph_a
        protograph_a = abstract.Protograph(protograph_a)
        protograph_b = abstract.Protograph(protograph_b)
        field = protograph_a.field.order

        # identify the number of qudits in each sector
        self.sector_size = protograph_a.group.lift_dim * np.outer(
            protograph_a.shape[::-1],
            protograph_b.shape[::-1],
        )
        qudits_to_conjugate = slice(self.sector_size[0, 0], None) if conjugate else None

        # identify sub-matrices and their transposes
        matrix_a = protograph_a.matrix
        matrix_b = protograph_b.matrix
        matrix_a_T = protograph_a.T.matrix
        matrix_b_T = protograph_b.T.matrix

        # construct the nontrivial blocks in the matrix
        mat_H1_In2 = np.kron(matrix_a, np.eye(matrix_b.shape[1], dtype=int))
        mat_In1_H2 = np.kron(np.eye(matrix_a.shape[1], dtype=int), matrix_b)
        mat_H1_Im2_T = np.kron(matrix_a_T, np.eye(matrix_b.shape[0], dtype=int))
        mat_Im1_H2_T = np.kron(np.eye(matrix_a.shape[0], dtype=int), matrix_b_T)

        # construct the parity check matrices
        matrix_x = abstract.Protograph(np.block([mat_H1_In2, -mat_Im1_H2_T])).lift()
        matrix_z = abstract.Protograph(np.block([mat_In1_H2, mat_H1_Im2_T])).lift()
        CSSCode.__init__(
            self, matrix_x, matrix_z, field, conjugate=qudits_to_conjugate, skip_validation=True
        )


################################################################################
# classical and quantum Tanner codes


class TannerCode(ClassicalCode):
    """Classical Tanner code, as described in DOI:10.1109/TIT.1981.1056404.

    A Tanner code T(G,C) is constructed from:
    [1] A bipartite "half-regular" graph G.  That is, a graph...
        ... with two sets of nodes, V and W.
        ... in which all nodes in W have degree n.
    [2] A classical code C on n bits.

    For convenience, we make G directed, with edges directed from V to W.  The node sets V and W can
    then be identified, respectively, by the sources and sinks of G.

    The Tanner code T(G,C) is defined on |W| bits.  A |W|-bit string x is a code word of T(G,C) iff,
    for every node v in V, the bits of x incident to v are a code word of C.

    This construction requires an ordering the edges E(v) adjacent to each vertex v.  This class
    sorts E(v) by the value of the "sort" attribute attached to each edge.  If there is no "sort"
    attribute, its value is treated as corresponding neighbor of v.

    Notes:
    - If the subcode C has m checks, its parity matrix has shape (m,n).
    - The code T(G,C) has |W| bits and |V|m checks.
    """

    subgraph: nx.DiGraph
    subcode: ClassicalCode

    def __init__(self, subgraph: nx.DiGraph, subcode: ClassicalCode) -> None:
        """Construct a classical Tanner code."""
        self.subgraph = subgraph
        self.subcode = subcode
        sources = [node for node in subgraph if subgraph.in_degree(node) == 0]
        sinks = [node for node in subgraph if subgraph.out_degree(node) == 0]
        sink_indices = {sink: idx for idx, sink in enumerate(sorted(sinks))}

        num_bits = len(sinks)
        num_checks = len(sources) * subcode.num_checks
        matrix = np.zeros((num_checks, num_bits), dtype=int)
        for idx, source in enumerate(sorted(sources)):
            checks = range(subcode.num_checks * idx, subcode.num_checks * (idx + 1))
            bits = [sink_indices[sink] for sink in self._get_sorted_neighbors(source)]
            matrix[np.ix_(checks, bits)] = subcode.matrix
        ClassicalCode.__init__(self, matrix, subcode.field.order)

    def _get_sorted_neighbors(self, node: object) -> Sequence[object]:
        """Sorted neighbors of the given node."""
        return sorted(
            self.subgraph.neighbors(node),
            key=lambda neighbor: self.subgraph[node][neighbor].get("sort", neighbor),
        )


class QTCode(CSSCode):
    """Quantum Tanner code: a CSS code for qudits defined on the faces of a Cayley complex

    Altogether, a quantum Tanner code is defined by:
    - two symmetric (self-inverse) subsets A and B of a group G, and
    - two classical codes C_A and C_B, respectively with block lengths |A| and |B|.

    The X-type parity checks of a quantum Tanner code are the checks of a classical Tanner code
    whose generating graph is the subgraph_0 of the Cayley complex (A,B).  The subcode of this
    classical Tanner code is ~(C_A ⊗ C_B), where ~C is the dual code to C.

    The Z-type parity checks are similarly defined with subgraph_1 and subcode ~(~C_A ⊗ ~C_B).

    Notes:
    - "Good" quantum Tanner code: projective special linear group and random classical codes.

    References:
    - https://errorcorrectionzoo.org/c/quantum_tanner
    - https://arxiv.org/abs/2206.07571
    - https://arxiv.org/abs/2202.13641
    """

    complex: CayleyComplex

    def __init__(
        self,
        subset_a: Collection[abstract.GroupMember],
        subset_b: Collection[abstract.GroupMember],
        code_a: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_b: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        field: int | None = None,
        *,
        conjugate: slice | Sequence[int] | None = (),
    ) -> None:
        """Construct a quantum Tanner code."""
        if code_b is None:
            code_b = code_a
        code_a = ClassicalCode(code_a, field)
        code_b = ClassicalCode(code_b, field)
        if field is None and code_a.field is not code_b.field:
            raise ValueError("The sub-codes provided for this QTCode are over different fields")

        self.complex = CayleyComplex(subset_a, subset_b)
        assert code_a.num_bits == len(self.complex.subset_a)
        assert code_b.num_bits == len(self.complex.subset_b)

        subcode_x = ~ClassicalCode.tensor_product(code_a, code_b)
        subcode_z = ~ClassicalCode.tensor_product(~code_a, ~code_b)
        matrix_x = TannerCode(self.complex.subgraph_0, subcode_x).matrix
        matrix_z = TannerCode(self.complex.subgraph_1, subcode_z).matrix
        CSSCode.__init__(self, matrix_x, matrix_z, field, conjugate=conjugate, skip_validation=True)
