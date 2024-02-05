"""Methods for generation of random codes, classical and quantum.

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

# import itertools
# import time
import galois

# from typing import TYPE_CHECKING, Literal

import numpy as np
import qldpc
from qldpc.codes import ClassicalCode, QTCode

DEFAULT_DISTANCE_TRIALS = 10


def get_distance(
    self,
    *,
    num_trials: int = DEFAULT_DISTANCE_TRIALS,
    vector: galois.FieldArray | None = None,
    exact: bool = False,
    brute: bool = True,
    **decoder_args: object,
) -> int:
    """Find smallest Hamming distance between input vector and a codeword.

    This method solves the same optimization problem as in CSSCode.get_one_distance_upper_bound
    """
    if brute:
        return self.distance_bruteforce(vector)

    if vector is None:
        vector = self.field.Zeros(self.num_bits)
    else:
        assert vector.shape == (self.num_bits, 1)
        vector = self.field(vector)

    syndrome = self.matrix @ vector
    effective_syndrome = np.append(syndrome, [1])
    dist_bound = np.inf

    for _ in range(num_trials):
        word = self.field.Random(self.num_bits)
        effective_check_matrix = np.vstack([self.matrix, word]).view(np.ndarray)

        closest_vec = qldpc.decoder.decode(
            effective_check_matrix,
            effective_syndrome,
            exact=exact,
            modulus=self.field.order,
            **decoder_args,
        )
    dist_bound = min(dist_bound, np.count_nonzero(closest_vec))

    return dist_bound


def get_distance(
    self,
    pauli: Literal[Pauli.X, Pauli.Z] | None = None,
    *,
    lower: bool = False,
    upper: int | None = None,
    vector: galois.FieldArray | None = None,
    **decoder_args: object,
) -> int:
    """Least possible distance between the input vector and any nontrivial logical operator

    If vector is None, then it initializes to the zero vector and the function computes the
    distance of this code, i.e., minimum weight of a nontrivial logical operator .

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
            self.get_distance(Pauli.X, lower=lower, upper=upper, vector=vector, **decoder_args),
            self.get_distance(Pauli.Z, lower=lower, upper=upper, vector=vector, **decoder_args),
        )

    if lower:
        return self.get_distance_lower_bound(pauli)

    if upper is not None:
        return self.get_distance_upper_bound(
            pauli, num_trials=upper, vector=vector, **decoder_args
        )

    return self.get_distance_exact(pauli, **decoder_args)

def get_distance_lower_bound(
    self, pauli: Literal[Pauli.X, Pauli.Z], vector: galois.FieldArray | None = None
) -> int:
    """Lower bound to the X-distance or Z-distance of this code."""
    assert pauli == Pauli.X or pauli == Pauli.Z
    pauli = pauli if not self._codes_equal else Pauli.X
    return (
        self.code_z.get_distance(vector=vector)
        if pauli == Pauli.X
        else self.code_x.get_distance(vector=vector)
    )

def get_distance_upper_bound(
    self,
    pauli: Literal[Pauli.X, Pauli.Z],
    num_trials: int,
    vector: galois.FieldArray | None = None,
    **decoder_args: object,
) -> int:
    """Upper bound to the X-distance or Z-distance of this code, minimized over many trials.

    All keyword arguments are passed to `CSSCode.get_one_distance_upper_bound`.
    """
    assert pauli == Pauli.X or pauli == Pauli.Z
    return min(
        self.get_one_distance_upper_bound(pauli, vector=vector, **decoder_args)
        for _ in range(num_trials)
    )

# TODO: Modify to take any x
def get_one_distance_upper_bound(
    self,
    pauli: Literal[Pauli.X, Pauli.Z],
    vector: galois.FieldArray | None = None,
    **decoder_args: object,
) -> int:
    """Single upper bound to the least possible distance between the input vector
    and any nontrivial logical operator.

    If vector is None, then it initializes to the zero vector and the function computes the
    distance of this code, i.e., minimum weight of a nontrivial logical (X or Z) operator .

    This method uses a randomized algorithm described in arXiv:2308.07915 (and also below).

    Args:
        pauli: Pauli operator choosing whether to compute an X-distance or Z-distance bound.
        decoder_args: Keyword arguments are passed to a decoder in `decode`.
        vector: The vector for which distance needs to be computed. 
                Setting it to None gives code distance.
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
    if self._field_order != 2:
        raise ValueError(
            "Distance upper bound calculation not implemented for fields of order > 2"
        )

    # define code_z and pauli_z as if we are computing X-distance
    code_z = self.code_z if pauli == Pauli.X else self.code_x
    pauli_z: Literal[Pauli.Z, Pauli.X] = Pauli.Z if pauli == Pauli.X else Pauli.X

    # construct the effective syndrome
    if vector is None:
        vector = self.field.Zeros(code_z.num_bits)
    else:
        assert vector.shape == (code_z.num_bits, 1)
        vector = self.field(vector)
    syndrome = code_z.matrix @ vector
    effective_syndrome = np.append(syndrome, [1])

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
def get_distance_exact(self, pauli: Literal[Pauli.X, Pauli.Z], **decoder_args: object) -> int:
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

    # minimize the weight of logical X-type or Z-type operators
    for logical_qubit_index in range(self.dimension):
        self._minimize_weight_of_logical_op(pauli, logical_qubit_index, **decoder_args)

    # return the minimum weight of logical X-type or Z-type operators
    return np.count_nonzero(self.get_logical_ops()[pauli.index].view(np.ndarray), axis=-1).min()
