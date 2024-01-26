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
import warnings

import cvxpy
import ldpc
import numpy as np
import numpy.typing as npt
import pymatching


def decode_with_BP_OSD(
    matrix: npt.NDArray[np.int_],
    syndrome: npt.NDArray[np.int_],
    **decoder_args: object,
) -> npt.NDArray[np.int_]:
    """Decode with belief propagation with ordered statistics (BP+OSD).

    For details about the BD-OSD decoder and its arguments, see:
    - Documentation: https://roffe.eu/software/ldpc/ldpc/osd_decoder.html
    - Reference: https://arxiv.org/pdf/2005.07016.pdf
    """
    bposd_decoder = ldpc.bposd_decoder(
        matrix, osd_order=decoder_args.pop("osd_order", 0), **decoder_args
    )
    return bposd_decoder.decode(syndrome)


def decode_with_MWPM(
    matrix: npt.NDArray[np.int_],
    syndrome: npt.NDArray[np.int_],
    **decoder_args: object,
) -> npt.NDArray[np.int_]:
    """Decode with minimum weight perfect matching (MWPM)."""
    matching = pymatching.Matching.from_check_matrix(matrix, **decoder_args)
    return matching.decode(syndrome)


def decode_with_ILP(
    matrix: npt.NDArray[np.int_],
    syndrome: npt.NDArray[np.int_],
    **decoder_args: object,
) -> npt.NDArray[np.int_]:
    """Decode with an integer linear program (ILP).

    Supports integers modulo q for q > 2 with a "modulus" argument.
    All remaining keyword arguments are passed to `cvxpy.Problem.solve`.
    """
    modulus = int(decoder_args.pop("modulus", 2))  # type:ignore[call-overload]
    if modulus < 2:
        raise ValueError(f"Decoding problems must have modulus >= 2 (provided modulus: {modulus}")
    if modulus == 2:
        opt_variables = cvxpy.Variable(matrix.shape[1], boolean=True)
    else:
        opt_variables = cvxpy.Variable(matrix.shape[1], integer=True)
    objective = cvxpy.Minimize(sum(iter(opt_variables)))

    # collect constraints, using boolean slack variables to relax each constraint of the form
    # `expression = val mod q` to `expression = val + sum_j q^j s_j`
    constraints = []
    matrix = matrix % modulus
    syndrome = syndrome % modulus
    for check, syndrome_bit in zip(matrix, syndrome):
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
            zero_mod_q = 0  # pragma: no cover
        constraints.append(check @ opt_variables == syndrome_bit + zero_mod_q)

    if modulus > 2:
        constraints += [var >= 0 for var in iter(opt_variables)]
        constraints += [var <= modulus for var in iter(opt_variables)]

    # solve the optimization problem!
    problem = cvxpy.Problem(objective, constraints)
    result = problem.solve(**decoder_args)
    if not isinstance(result, float) or not np.isfinite(result):
        message = "Optimal solution to integer linear program could not be found!"
        raise ValueError(message + f"\nSolver output: {result}")
    solution = problem.variables()[0].value
    return np.array(solution).astype(int) % modulus


def decode(
    matrix: npt.NDArray[np.int_],
    syndrome: npt.NDArray[np.int_],
    *,
    exact: bool = False,
    **decoder_args: object,
) -> npt.NDArray[np.int_]:
    """Find a `vector` that solves `matrix @ vector == syndrome mod 2`.

    - If passed an explicit decoder, use it.
    - If no decoder is provided and `exact is True`, solve exactly with an integer linear program.
    - Otherwise, use a BP-OSD decoder.

    In all cases, pass the `decoder_args` to the decoder that is used.
    """
    if callable(custom_decoder := decoder_args.pop("decoder", None)):
        if exact:
            warnings.warn(
                "Exact decoding was reqested, but cannot be guaranteed with a custom decoder"
            )
        return custom_decoder(matrix, syndrome, **decoder_args)

    if not exact:
        return decode_with_BP_OSD(matrix, syndrome, **decoder_args)

    return decode_with_ILP(matrix, syndrome, **decoder_args)
