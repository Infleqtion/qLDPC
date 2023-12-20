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
import abc
import functools
import itertools
from typing import Collection, Literal, Optional, Sequence, Union

import cachetools
import ldpc.code_util
import ldpc.codes
import ldpc.mod2
import networkx as nx
import numpy as np
import numpy.typing as npt

import qldpc
from qldpc import abstract
from qldpc.objects import CayleyComplex, Node, Pauli

IntegerMatrix = npt.NDArray[np.int_] | Sequence[Sequence[int]]
ObjectMatrix = npt.NDArray[np.object_] | Sequence[Sequence[object]]


################################################################################
# template error correction code classes


class AbstractCode(abc.ABC):
    """Template class for error-correcting codes."""

    @abc.abstractproperty
    def matrix(self) -> npt.NDArray[np.int_]:
        """Parity check matrix of this code."""

    @functools.cached_property
    def graph(self) -> nx.DiGraph:
        """Tanner graph of this code."""
        return self.matrix_to_graph(self.matrix)

    @classmethod
    @abc.abstractmethod
    def matrix_to_graph(cls, matrix: IntegerMatrix) -> nx.DiGraph:
        """Convert a parity check matrix into a Tanner graph."""

    @classmethod
    @abc.abstractmethod
    def graph_to_matrix(cls, graph: nx.DiGraph) -> npt.NDArray[np.int_]:
        """Convert a Tanner graph into a parity check matrix."""


class BitCode(AbstractCode):
    """Classical (binary, linear) error-correcting code.

    A classical binary code C = {x} is a set of bitstings x, called code words.  We consider only
    linear codes here, for which any linear combination of code words is also code word.

    Operationally, we define a classical code by a parity check matrix H with dimensions
    (num_checks, num_bits).  Each row of H represents a linear constraint (a "check") that code
    words must satisfy.  A bitstring x is a code word iff H @ x = 0 mod 2.
    """

    def __init__(self, matrix: Union["BitCode", IntegerMatrix]) -> None:
        """Construct a classical code from a parity check matrix."""
        if isinstance(matrix, BitCode):
            self._matrix = matrix.matrix
        else:
            self._matrix = np.array(matrix)

    @property
    def matrix(self) -> npt.NDArray[np.int_]:
        """Parity check matrix of this code."""
        return self._matrix

    @classmethod
    def matrix_to_graph(cls, matrix: IntegerMatrix) -> nx.DiGraph:
        """Convert a parity check matrix H into a Tanner graph.

        The Tanner graph is a bipartite graph with (num_checks, num_bits) vertices, respectively
        identified with the checks and bits of the code.  The check vertex c and the bit vertex b
        share an edge iff c addresses b; that is, edge (c, b) is in the graph iff H[c, b] == 1.
        """
        edges = [
            (Node(index=int(row), is_data=False), Node(index=int(col), is_data=True))
            for row, col in zip(*np.where(matrix))
        ]
        return nx.DiGraph(edges)

    @classmethod
    def graph_to_matrix(cls, graph: nx.DiGraph) -> npt.NDArray[np.int_]:
        """Convert a Tanner graph into a parity check matrix."""
        num_bits = sum(1 for node in graph.nodes() if node.is_data)
        num_checks = len(graph.nodes()) - num_bits
        matrix = np.zeros((num_checks, num_bits), dtype=int)
        for check_node, bit_node in graph.edges():
            matrix[check_node.index, bit_node.index] = 1
        return matrix

    @functools.cached_property
    def generator(self) -> npt.NDArray[np.int_]:
        """Generator of this code.

        The generator of a code C is a matrix whose rows form a basis for C.
        """
        return ldpc.code_util.construct_generator_matrix(self._matrix)

    def words(self) -> npt.NDArray[np.int_]:
        """Code words of this code."""
        return ldpc.code_util.codewords(self._matrix)

    def get_random_word(self) -> npt.NDArray[np.int_]:
        """Random code word: a sum all generators with random (0/1) coefficients."""
        return np.random.randint(2, size=self.generator.shape[0]) @ self.generator % 2

    def dual(self) -> "BitCode":
        """Dual to this code.

        The dual code ~C is the set of bitstrings orthogonal to C:
        ~C = { x : x @ y = 0 for all y in C }.
        The parity check matrix of ~C is equal to the generator of C.
        """
        return BitCode(self.generator)

    def __invert__(self) -> "BitCode":
        """Dual to this code."""
        return self.dual()

    @classmethod
    def tensor_product(cls, code_a: "BitCode", code_b: "BitCode") -> "BitCode":
        """Tensor product C_a ⊗ C_b of two codes C_a and C_b.

        Let G_a and G_b respectively denote the generators C_a and C_b.
        Definition: C_a ⊗ C_b is the code whose generators are G_a ⊗ G_b.

        G_a ⊗ G_b is the check matrix of ~(C_a ⊗ C_b).
        We therefore construct ~(C_a ⊗ C_b) and return its dual ~~(C_a ⊗ C_b) = C_a ⊗ C_b.
        """
        generator_ab = np.kron(code_a.generator, code_b.generator)
        return ~BitCode(generator_ab)

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
        return ldpc.mod2.rank(self._matrix)

    @property
    def num_logical_bits(self) -> int:
        """The number of logical bits encoded by this code."""
        return self.num_bits - self.rank

    @functools.cache
    def get_distance(self) -> int:
        """The distance of this code."""
        return ldpc.code_util.compute_code_distance(self._matrix)

    def get_code_params(self) -> tuple[int, int, int]:
        """Compute the parameters of this code: [n,k,d].

        Here:
        - n is the number of data bits
        - k is the number of encoded ("logical") bits
        - d is the code distance
        """
        return self.num_bits, self.num_logical_bits, self.get_distance()

    @classmethod
    def random(cls, bits: int, checks: int) -> "BitCode":
        """Construct a random classical code with the given number of bits and checks."""
        rows, cols = checks, bits
        matrix = np.random.randint(2, size=(rows, cols))
        for row in range(matrix.shape[0]):
            if not matrix[row, :].any():
                matrix[row, np.random.randint(cols)] = 1  # pragma: no cover
        for col in range(matrix.shape[1]):
            if not matrix[:, col].any():
                matrix[np.random.randint(rows), col] = 1  # pragma: no cover
        return BitCode(matrix)

    @classmethod
    def repetition(cls, num_bits: int) -> "BitCode":
        """Construct a repetition code on the given number of bits."""
        return BitCode(ldpc.codes.rep_code(num_bits))

    @classmethod
    def ring(cls, num_bits: int) -> "BitCode":
        """Construct a repetition code with periodic boundary conditions."""
        return BitCode(ldpc.codes.ring_code(num_bits))

    @classmethod
    def hamming(cls, rank: int) -> "BitCode":
        """Construct a hamming code of a given rank."""
        return BitCode(ldpc.codes.hamming_code(rank))


class QubitCode(AbstractCode):
    """Template class for a qubit-based quantum error-correcting code.

    The parity check matrix of a qubit code is organized into an array with dimensions
    (num_checks, 2, num_qubits).  The first and last dimensions respectively index a stabilizer and
    a qubit, while the middle dimension indexes whether a given stabilizer addresses a given qubit
    with an X-type or Z-type Pauli operator.  If a stabilizer S addresses qubit q with a Pauli-Y
    operator, that is treated as S addressing q with both Pauli-X and Pauli-Z.

    The Tanner graph of a qubit code nearly identical to that of a classical code, with the only
    difference being that the edge (c, b) is now tagged by (i.e., has an attribute to indicate) the
    Pauli operator with which check qubit c addresses data qubit b.
    """

    @classmethod
    def matrix_to_graph(cls, matrix: IntegerMatrix) -> nx.DiGraph:
        """Convert a parity check matrix into a Tanner graph."""
        graph = nx.DiGraph()
        for row, col_xz, col in zip(*np.where(matrix)):
            node_check = Node(index=int(row), is_data=False)
            node_qubit = Node(index=int(col), is_data=True)
            graph.add_edge(node_check, node_qubit)
            pauli = Pauli.X if col_xz == Pauli.X.index else Pauli.Z
            old_pauli = graph[node_check][node_qubit].get(Pauli, Pauli.I)
            graph[node_check][node_qubit][Pauli] = old_pauli * pauli
        return graph

    @classmethod
    def graph_to_matrix(cls, graph: nx.DiGraph) -> nx.DiGraph:
        """Convert a Tanner graph into a parity check matrix."""
        num_qubits = sum(1 for node in graph.nodes() if node.is_data)
        num_checks = len(graph.nodes()) - num_qubits
        matrix = np.zeros((num_checks, 2, num_qubits), dtype=int)
        for (check_node, bit_node), pauli in nx.get_edge_attributes(graph, Pauli).items():
            matrix[check_node.index, pauli.index, bit_node.index] = 1
        return matrix


class CSSCode(QubitCode):
    """CSS qubit code, with separate X-type and Z-type parity checks.

    In order for the X-type and Z-type parity checks to be "compatible", the X-type stabilizers must
    commute with the Z-type stabilizers.  Mathematically, this requirement can be written as

    H_x @ H_z.T == 0,

    where H_x and H_z are, respectively, the parity check matrices of the classical codes that
    define the X-type and Z-type stabilizers of the CSS code.

    A CSSCode can additionally specify which data qubits should be hadamard-transformed before/after
    syndrome extraction, thereby transforming the operators that address a specified data qubit as
    (X,Y,Z) <--> (Z,Y,X).

    For completion, a CSSCode can also "shift" the Pauli operators on a qubit, moving vertically
    along the following table:

    ―――――――――――
    | XY | YX |
    ―――――――――――
    | XZ | ZX |
    ―――――――――――
    | YZ | ZY |
    ―――――――――――

    Qubit shifts are specified by a dictionary mapping qubit_index --> shift_index, where
    shift_index = +1, 0, and -1 mod 3 respectively refer to the top, middle, and bottom rows of the
    table.

    Physically, a shift of +1 and -1 respectively correspond to conjugating the Pauli operators
    addressing a qubit by sqrt(X) and sqrt(Z) rotations.
    """

    code_x: BitCode  # X-type parity checks, measuring Z-type errors
    code_z: BitCode  # Z-type parity checks, measuring X-type errors
    conjugate: Optional[slice | Sequence[int]]
    shifts: Optional[dict[int, int]]
    self_dual: bool

    _logical_ops: Optional[npt.NDArray[np.int_]] = None

    def __init__(
        self,
        code_x: BitCode | IntegerMatrix,
        code_z: BitCode | IntegerMatrix,
        qubits_to_conjugate: Optional[slice | Sequence[int]] = None,
        qubit_shifts: Optional[dict[int, int]] = None,
        self_dual: bool = False,
    ) -> None:
        """Construct a CSS code from X-type and Z-type parity checks."""
        self.code_x = BitCode(code_x)
        self.code_z = BitCode(code_z)
        self.conjugate = qubits_to_conjugate
        self.shifts = qubit_shifts
        self.self_dual = self_dual

        assert self.code_x.matrix.ndim == self.code_z.matrix.ndim == 2
        assert self.code_x.num_bits == self.code_z.num_bits
        assert not np.any(self.code_x.matrix @ self.code_z.matrix.T % 2)

    @functools.cached_property
    def matrix(self) -> npt.NDArray[np.int_]:
        """Overall parity check matrix."""
        matrix = np.block(
            [
                [np.zeros_like(self.code_z.matrix), self.code_z.matrix],
                [self.code_x.matrix, np.zeros_like(self.code_x.matrix)],
            ]
        ).reshape((self.num_checks, 2, -1))

        if self.conjugate:
            # swap X and Z operators on the qubits to conjugate
            matrix[:, :, self.conjugate] = np.roll(matrix[:, :, self.conjugate], 1, axis=1)

        if self.shifts:
            # identify qubits to shift up or down along the table of Pauli pairs
            shifts_up = tuple(qubit for qubit, shift in self.shifts.items() if shift % 3 == 1)
            shifts_dn = tuple(qubit for qubit, shift in self.shifts.items() if shift % 3 == 2)
            # For qubits shifting up, any stabilizer addressing a qubit with a Z should now also
            # address that qubit with X, so copy Z to X.  Likewise for shifting down and X to Z.
            matrix[:, Pauli.X.index, shifts_up] |= matrix[:, Pauli.Z.index, shifts_up]
            matrix[:, Pauli.Z.index, shifts_dn] |= matrix[:, Pauli.X.index, shifts_dn]

        return matrix

    @property
    def num_checks(self) -> int:
        """Number of parity checks."""
        return self.code_x.matrix.shape[0] + self.code_z.matrix.shape[0]

    @property
    def num_qubits(self) -> int:
        """Number of data qubits in this code."""
        return self.code_x.matrix.shape[1]

    @property
    def num_logical_qubits(self) -> int:
        """Number of logical qubits encoded by this code."""
        return self.code_x.num_logical_bits + self.code_z.num_logical_bits - self.num_qubits

    def get_code_params(
        self, *, lower: bool = False, upper: Optional[int] = None, **decoder_args: object
    ) -> tuple[int, int, int]:
        """Compute the parameters of this code: [[n,k,d]].

        Here:
        - n is the number of data qubits
        - k is the number of encoded ("logical") qubits
        - d is the code distance

        Keyword arguments are passed to the calculation of code distance.
        """
        distance = self.get_distance(pauli=None, lower=lower, upper=upper, **decoder_args)
        return self.num_qubits, self.num_logical_qubits, distance

    def get_distance(
        self,
        pauli: Optional[Literal[Pauli.X, Pauli.Z]] = None,
        *,
        lower: bool = False,
        upper: Optional[int] = None,
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
        programming.  Warning: this is an exponentially scaling (NP-complete) problem.

        All remaining keyword arguments are passed to a decoder in `decode`.
        """
        assert pauli in [None, Pauli.X, Pauli.Z]
        assert lower is False or upper is None
        if pauli is None:
            # minimize over X-distance and Z-distance
            return min(
                self.get_distance(Pauli.X, lower=lower, upper=upper, **decoder_args),
                self.get_distance(Pauli.Z, lower=lower, upper=upper, **decoder_args),
            )

        if lower:
            # distance of the Z-type subcode correcting X-type errors, or vice versa
            return self.code_z.get_distance() if pauli == Pauli.X else self.code_x.get_distance()

        if upper is not None:
            # compute an upper bound to the distance with a decoder
            return self.get_distance_upper_bound(pauli, upper, **decoder_args)

        # exact distance with an integer linear program
        return self.get_distance_exact(pauli, **decoder_args)

    @functools.cache
    def get_distance_upper_bound(
        self,
        pauli: Literal[Pauli.X, Pauli.Z],
        num_trials: int,
        *,
        ensure_nontrivial: bool = True,
        **decoder_args: object,
    ) -> int:
        """Upper bound to the X-distance or Z-distance of this code, minimized over many trials.

        All keyword arguments are passed to `CSSCode.get_one_distance_upper_bound`.
        """
        return min(
            self.get_one_distance_upper_bound(
                pauli, ensure_nontrivial=ensure_nontrivial, **decoder_args
            )
            for _ in range(num_trials)
        )

    def get_one_distance_upper_bound(
        self,
        pauli: Literal[Pauli.X, Pauli.Z],
        *,
        ensure_nontrivial: bool = True,
        **decoder_args: object,
    ) -> int:
        """Single upper bound to the X-distance or Z-distance of this code.

        This method uses a randomized algorithm described in arXiv:2308.07915 (and also below).

        Args:
            pauli: Pauli operator choosing whether to compute an X-distance or Z-distance bound.
            ensure_nontrivial: When we generate a random logical operator, should we ensure that
                this operator is nontrivial?  (default: True)
            decoder_args: Keyword arguments are passed to a decoder in `decode`.
        Returns:
            An upper bound on the X-distance or Z-distance.

        For ease of language, we henceforth assume that we are computing an X-distance.

        Pick a random Z-type logical operator Z(w_z) whose support is indicated by the bistring w_z.
        If `ensure_nontrivial is True`, ensure that Z(w_z) is a nontrivial logical operator,
        although doing so is not strictly necessary.

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
        # define code_z and pauli_z as if we are computing X-distance
        code_z = self.code_z if pauli == Pauli.X else self.code_x
        pauli_z: Literal[Pauli.Z, Pauli.X] = Pauli.Z if pauli == Pauli.X else Pauli.X

        # construct the effective syndrome
        effective_syndrome = np.zeros(code_z.num_checks + 1, dtype=int)
        effective_syndrome[-1] = 1

        logical_op_found = False
        while not logical_op_found:
            # support of pauli string with a trivial syndrome
            word = self.get_random_logical_op(pauli_z, ensure_nontrivial=ensure_nontrivial)

            # support of a candidate pauli-type logical operator
            effective_check_matrix = np.vstack([code_z.matrix, word])
            candidate_logical_op = qldpc.decode(
                effective_check_matrix, effective_syndrome, exact=False, **decoder_args
            )

            # check whether the decoding was successful
            actual_syndrome = effective_check_matrix @ candidate_logical_op % 2
            logical_op_found = np.array_equal(actual_syndrome, effective_syndrome)

        # return the Hamming weight of the logical operator
        return candidate_logical_op.sum()

    @cachetools.cached(cache={}, key=lambda self, pauli, **decoder_args: (self, pauli))
    def get_distance_exact(self, pauli: Literal[Pauli.X, Pauli.Z], **decoder_args: object) -> int:
        """Exact X-distance or Z-distance of this code."""
        # minimize the weight of logical X-type or Z-type operators
        for logical_qubit_index in range(self.num_logical_qubits):
            self._minimize_weight_of_logical_op(pauli, logical_qubit_index, **decoder_args)

        # return the minimum weight of logical X-type or Z-type operators
        return self.get_logical_ops()[pauli.index].sum(-1).min()

    def get_logical_ops(self) -> npt.NDArray[np.int_]:
        """Complete basis of nontrivial X-type and Z-type logical operators for this code.

        Logical operators are represented by a three-dimensional array `logical_ops` with dimensions
        (2, k, n), where k and n are respectively the numbers of logical and physical qubits in this
        code.  The bitstring `logical_ops[0, 4, :]`, for example, indicates the support (i.e., the
        physical qubits addressed nontrivially) by the logical Pauli-X operator on logical qubit 4.

        Logical operators are identified using the symplectic Gram-Schmidt orthogonalization
        procedure described in arXiv:0903.5256.
        """
        if self._logical_ops is not None:
            return self._logical_ops

        # identify candidate X-type and Z-type operators
        candidates_x = list(self.code_z.generator)
        candidates_z = list(self.code_x.generator)

        # collect logical operators sequentially
        logicals_x = []
        logicals_z = []

        # iterate over all candidate X-type operators
        while candidates_x:
            op_x = candidates_x.pop()
            found_logical_pair = False

            # check whether op_x anti-commutes with any of the candidate Z-type operators
            for zz, op_z in enumerate(candidates_z):
                if op_x @ op_z % 2:
                    # op_x and op_z anti-commute, so they are conjugate pair of logical operators!
                    found_logical_pair = True
                    logicals_x.append(op_x)
                    logicals_z.append(op_z)
                    del candidates_z[zz]
                    break

            if found_logical_pair:
                # If any other candidate X-type operators anti-commute with op_z, it's because they
                # have an op_x component.  Remove that component.  Likewise with Z-type candidates.
                for xx, other_x in enumerate(candidates_x):
                    if other_x @ op_z % 2:
                        candidates_x[xx] = (other_x + op_x) % 2
                for zz, other_z in enumerate(candidates_z):
                    if other_z @ op_x % 2:
                        candidates_z[zz] = (other_z + op_z) % 2

        assert len(logicals_x) == self.num_logical_qubits
        self._logical_ops = np.stack([logicals_x, logicals_z]).astype(int)
        return self._logical_ops

    def get_random_logical_op(
        self, pauli: Literal[Pauli.X, Pauli.Z], ensure_nontrivial: bool
    ) -> npt.NDArray[np.int_]:
        """Return a random logical operator of a given type.

        A random logical operator may be trivial, which is to say that it may be equal to the
        identity modulo stabilizers.  If `ensure_nontrivial is True`, ensure that the logical
        operator we return is nontrivial.
        """
        if ensure_nontrivial:
            random_logical_qubit_index = np.random.randint(self.num_logical_qubits)
            return self.get_logical_ops()[pauli.index, random_logical_qubit_index]
        return (self.code_z if pauli == Pauli.X else self.code_x).get_random_word()

    def _minimize_weight_of_logical_op(
        self,
        pauli: Literal[Pauli.X, Pauli.Z],
        logical_qubit_index: int,
        **decoder_args: object,
    ) -> None:
        """Minimize the weight of a logical operator.

        This method solves the same optimization problem as in CSSCode.get_one_distance_upper_bound,
        but exactly with integer linear programming (which has exponential complexity).
        """
        assert 0 <= logical_qubit_index < self.num_logical_qubits
        code = self.code_z if pauli == Pauli.X else self.code_x
        word = self.get_logical_ops()[(~pauli).index, logical_qubit_index]
        effective_check_matrix = np.vstack([code.matrix, word])
        effective_syndrome = np.zeros((code.num_checks + 1), dtype=int)
        effective_syndrome[-1] = 1
        logical_op = qldpc.decode(
            effective_check_matrix, effective_syndrome, exact=True, **decoder_args
        )
        assert self._logical_ops is not None
        self._logical_ops[pauli.index, logical_qubit_index] = logical_op


################################################################################
# bicycle and quasi-cyclic codes


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
        matrix_a: IntegerMatrix,
        matrix_b: Optional[IntegerMatrix] = None,
        *,
        conjugate: bool = False,
    ) -> None:
        """Construct a generalized bicycle code."""
        if matrix_b is None:
            matrix_b = matrix_a  # pragma: no cover
        matrix_a = np.array(matrix_a)
        matrix_b = np.array(matrix_b)
        assert np.array_equal(matrix_a @ matrix_b.T % 2, matrix_b.T @ matrix_a % 2)

        matrix_x = np.block([matrix_a, matrix_b.T])
        matrix_z = np.block([matrix_b, matrix_a.T])
        qubits_to_conjugate = slice(matrix_a.shape[0], None) if conjugate else None
        CSSCode.__init__(self, matrix_x, matrix_z, qubits_to_conjugate, self_dual=True)


class QCCode(GBCode):
    """Quasi-cyclic (QC) code.

    Inspired by arXiv:2308.07915.

    A quasi-cyclic code is a CSS code with parity check matrices
    - matrix_x = [A, B.T], and
    - matrix_z = [B, A.T],
    where A and B are block matrices identified with elements of a multivariate polynomial ring.
    Specifically, we can expand (say) A = sum_{i,j} A_{ij} x_i^j, where A_{ij} are (binary)
    coefficients and each x_i is the generator of a cyclic group of order R_i.

    A quasi-cyclic code is defined by...
    [1] sequence (R_0, R_1, ...) of cyclic group orders (one per variable, x_i), and
    [2] a list of nonzero terms in A and B, with the term x_i^j identified by the tuple (i, j).
    The polynomial A = x + y^3 + z^2, for example, is identified by [(0, 1), (1, 3), (2, 2)].
    """

    def __init__(
        self,
        dims: Sequence[int],
        terms_a: Collection[tuple[int, int]],
        terms_b: Optional[Collection[tuple[int, int]]] = None,
        *,
        conjugate: bool = False,
    ) -> None:
        """Construct a quasi-cyclic code."""
        if terms_b is None:
            terms_b = terms_a  # pragma: no cover

        groups = [abstract.CyclicGroup(dim) for dim in dims]
        group = abstract.Group.product(*groups)

        members_a = [group.generators[factor] ** power for factor, power in terms_a]
        members_b = [group.generators[factor] ** power for factor, power in terms_b]
        matrix_a = abstract.Element(group, *members_a).lift()
        matrix_b = abstract.Element(group, *members_b).lift()
        GBCode.__init__(self, matrix_a, matrix_b, conjugate=conjugate)


################################################################################
# hypergraph and lifted product codes


class HGPCode(CSSCode):
    """Hypergraph product (HGP) code.

    A hypergraph product code AB is constructed from two classical codes, A and B.

    Consider the following:
    - Code A has 3 data and 2 check bits.
    - Code B has 4 data and 3 check bits.
    We represent data (qu)bits by circles (○) and check (qu)bits by squares (□).

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
    - sector (0, 0) has 3×4=12 data qubits
    - sector (0, 1) has 3×3=9 check qubits
    - sector (1, 0) has 2×4=8 check qubits
    - sector (1, 1) has 2×3=6 data qubits

    Edges in G_AB are inherited across rows/columns from G_A and G_B.  For example, if rows r_1 and
    r_2 share an edge in G_A, then the same is true in every column of G_AB.

    By default, the check qubits in sectors (0, 1) of G_AB measure Pauli-Z operators.  Likewise with
    sector (1, 0) and Pauli-X operators.  If a HGP is constructed with `conjugate is True`, then the
    Pauli operators addressing the nodes in sector (1, 1) are switched.

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
        code_a: BitCode | IntegerMatrix,
        code_b: Optional[BitCode | IntegerMatrix] = None,
        *,
        conjugate: bool = False,
        self_dual: bool = False,
    ) -> None:
        """Construct a hypergraph product code."""
        if code_b is None:
            code_b = code_a
            self_dual = True
        code_a = BitCode(code_a)
        code_b = BitCode(code_b)

        # identify the number of qubits in each sector
        self.sector_size = np.outer(
            [code_a.num_bits, code_a.num_checks],
            [code_b.num_bits, code_b.num_checks],
        )

        # construct the parity check matrices of this code
        matrix_x, matrix_z, qubits_to_conjugate = HGPCode.get_hyper_product(
            code_a.matrix, code_b.matrix, conjugate=conjugate
        )
        CSSCode.__init__(self, matrix_x, matrix_z, qubits_to_conjugate, self_dual=self_dual)

    @classmethod
    def get_hyper_product(
        self,
        code_a: BitCode | IntegerMatrix,
        code_b: BitCode | IntegerMatrix,
        *,
        conjugate: bool = False,
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], Optional[slice]]:
        """Hypergraph product of two classical codes, as in arXiv:2202.01702.

        The parity check matrices of the hypergraph product code are:

        matrix_x = [H1 ⊗ In2, Im1 ⊗ H2.T]
        matrix_z = [In1 ⊗ H2, H1.T ⊗ Im2]

        Here (H1, H2) == (matrix_a, matrix_b), and I[m/n][1/2] are identity matrices,
        with (m1, n1) = H1.shape and (m2, n2) = H2.shape.

        If `conjugate is True`, we hadamard-transform the data qubits in sector (1, 1), which are
        addressed by the second block of matrix_x and marix_z above.
        """
        matrix_a = BitCode(code_a).matrix
        matrix_b = BitCode(code_b).matrix

        # construct the nontrivial blocks in the matrix
        mat_H1_In2 = np.kron(matrix_a, np.eye(matrix_b.shape[1], dtype=int))
        mat_In1_H2 = np.kron(np.eye(matrix_a.shape[1], dtype=int), matrix_b)
        mat_H1_Im2_T = np.kron(matrix_a.T, np.eye(matrix_b.shape[0], dtype=int))
        mat_Im1_H2_T = np.kron(np.eye(matrix_a.shape[0], dtype=int), matrix_b.T)

        # construct the parity check matrices
        matrix_x = np.block([mat_H1_In2, mat_Im1_H2_T])
        matrix_z = np.block([mat_In1_H2, mat_H1_Im2_T])
        qubits_to_conjugate = slice(mat_H1_In2.shape[1], None) if conjugate else None
        return matrix_x, matrix_z, qubits_to_conjugate

    @classmethod
    def get_graph_product(
        cls,
        graph_a: nx.DiGraph,
        graph_b: nx.DiGraph,
        *,
        conjugate: bool = False,
    ) -> nx.DiGraph:
        """Hypergraph product of two Tanner graphs."""
        graph_product = nx.cartesian_product(graph_a, graph_b)

        # fix edge orientation and tag each edge with a Pauli operator
        graph = nx.DiGraph()
        for node_fst, node_snd in graph_product.edges:
            # identify check vs. qubit nodes
            if node_fst[0].is_data == node_fst[1].is_data:
                node_qubit, node_check = node_fst, node_snd
            else:
                node_qubit, node_check = node_snd, node_fst
            graph.add_edge(node_check, node_qubit)

            # by default, this edge is Pauli-Z if the check qubit is in the (0, 1) sector
            pauli = Pauli.Z if node_check[0].is_data else Pauli.X
            # flip Z <--> X if `conjugate is True` and the data qubit is in the (1, 1) sector
            if conjugate and not node_qubit[0].is_data:
                pauli = ~pauli
            graph[node_check][node_qubit][Pauli] = pauli

        return nx.relabel_nodes(graph, HGPCode.get_product_node_map(graph_a.nodes, graph_b.nodes))

    @classmethod
    def get_product_node_map(
        cls, nodes_a: Collection[Node], nodes_b: Collection[Node]
    ) -> dict[tuple[Node, Node], Node]:
        """Map (dictionary) that re-labels nodes in the hypergraph product of two codes."""
        index_qubit = 0
        index_check = 0
        node_map = {}
        for node_a, node_b in itertools.product(sorted(nodes_a), sorted(nodes_b)):
            if node_a.is_data == node_b.is_data:
                # this is a data qubit in sector (0, 0) or (1, 1)
                node = Node(index=index_qubit, is_data=True)
                index_qubit += 1
            else:
                # this is a check qubit in sector (0, 1) or (1, 0)
                node = Node(index=index_check, is_data=False)
                index_check += 1
            node_map[node_a, node_b] = node
        return node_map


class LPCode(CSSCode):
    """Lifted product (LP) code.

    A lifted product code is essentially the same as a hypergraph product code, except that the
    parity check matrices are "protographs", or matrices whose entries are members of a group
    algebra over the field {0,1}.  Each of these entries can be "lifted" to a representation as
    square matrices of 0s and 1s, in which case the protograph is interpreted as a block matrix;
    this is called "lifting" the protograph.

    Notes:
    - A lifted product code with protographs of size 1×1 is a generalized bicycle code.

    References:
    - https://errorcorrectionzoo.org/c/lifted_product
    - https://arxiv.org/abs/2202.01702
    - https://arxiv.org/abs/2012.04068
    """

    def __init__(
        self,
        protograph_a: abstract.Protograph | ObjectMatrix,
        protograph_b: Optional[abstract.Protograph | ObjectMatrix] = None,
        *,
        conjugate: bool = False,
        self_dual: bool = False,
    ) -> None:
        """Construct a lifted product code."""
        if protograph_b is None:
            protograph_b = protograph_a
            self_dual = True
        protograph_a = abstract.Protograph(protograph_a)
        protograph_b = abstract.Protograph(protograph_b)

        # identify the number of qubits in each sector
        self.sector_size = protograph_a.group.lift_dim * np.outer(
            protograph_a.shape[::-1],
            protograph_b.shape[::-1],
        )

        # construct the parity check matrices of this code
        protograph_x, protograph_z, qubits_to_conjugate = LPCode.get_hyper_product(
            protograph_a, protograph_b, conjugate=conjugate
        )
        CSSCode.__init__(
            self,
            protograph_x.lift(),
            protograph_z.lift(),
            qubits_to_conjugate,
            self_dual=self_dual,
        )

    @classmethod
    def get_hyper_product(
        self,
        protograph_a: abstract.Protograph | ObjectMatrix,
        protograph_b: abstract.Protograph | ObjectMatrix,
        *,
        conjugate: bool = False,
    ) -> tuple[abstract.Protograph, abstract.Protograph, Optional[slice]]:
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
        protograph_a = abstract.Protograph(protograph_a)
        protograph_b = abstract.Protograph(protograph_b)
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
        protograph_x = abstract.Protograph(np.block([mat_H1_In2, mat_Im1_H2_T]))
        protograph_z = abstract.Protograph(np.block([mat_In1_H2, mat_H1_Im2_T]))

        # identify the qubits to conjugate
        if conjugate:
            sector_boundary = protograph_a.group.lift_dim * mat_H1_In2.shape[1]
            qubits_to_conjugate = slice(sector_boundary, None)
        else:
            qubits_to_conjugate = None

        return protograph_x, protograph_z, qubits_to_conjugate


################################################################################
# classical and quantum Tanner codes


class TannerCode(BitCode):
    """Classical Tanner code, as described in DOI:10.1109/TIT.1981.1056404.

    A Tanner code T(G,C) is constructed from:
    [1] A (k,n)-regular directed bipartite graph G.  That is, a graph...
        ... with two sets of nodes, V and W.
        ... in which all edges are directed from a node in V to a node in W.
        ... in which all nodes in V have degree k, and all nodes in W have degree n.
    [2] A classical code C on n bits.

    The Tanner code T(G,C) is defined on |W| bits.  A |W|-bit string x is a code word of T(G,C) iff,
    for every node v in V, the bits of x incident to v are a code word of C.

    This construction requires an ordering the edges E(v) adjacent to each vertex v.  This method
    sorts E(v) by the value of the "sort" attribute attached to each edge.  If there is no "sort"
    attribute, its value is treated as corresponding neighbor of v.

    Notes:
    - If the subcode C has m checks, its parity matrix has shape (m,n).
    - The code T(G,C) has |W| bits and |V|m checks.
    """

    subgraph: nx.DiGraph
    subcode: BitCode

    def __init__(self, subgraph: nx.DiGraph, subcode: BitCode) -> None:
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
            bits = [sink_indices[sink] for sink in self._get_sorted_children(source)]
            matrix[np.ix_(checks, bits)] = subcode.matrix
        BitCode.__init__(self, matrix)

    def _get_sorted_children(self, source: object) -> Sequence[object]:
        """Sorted children of the given source node."""
        return sorted(
            self.subgraph.neighbors(source),
            key=lambda neighbor: self.subgraph[source][neighbor].get("sort", neighbor),
        )


class QTCode(CSSCode):
    """Quantum Tanner code: a CSS code for qubits defined on the faces of a Cayley complex

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
        code_a: BitCode | IntegerMatrix,
        code_b: Optional[BitCode | IntegerMatrix] = None,
        *,
        rank: Optional[int] = None,
        conjugate: Sequence[int] = (),
        self_dual: bool = False,
    ) -> None:
        """Construct a quantum Tanner code."""
        if code_b is None:
            code_b = code_a
            self_dual = True
        code_a = BitCode(code_a)
        code_b = BitCode(code_b)
        self.complex = CayleyComplex(subset_a, subset_b, rank=rank)
        assert code_a.num_bits == len(self.complex.subset_a)
        assert code_b.num_bits == len(self.complex.subset_b)

        subcode_x = ~BitCode.tensor_product(code_a, code_b)
        subcode_z = ~BitCode.tensor_product(~code_a, ~code_b)
        matrix_x = TannerCode(self.complex.subgraph_0, subcode_x).matrix
        matrix_z = TannerCode(self.complex.subgraph_1, subcode_z).matrix
        CSSCode.__init__(self, matrix_x, matrix_z, conjugate, self_dual=self_dual)
