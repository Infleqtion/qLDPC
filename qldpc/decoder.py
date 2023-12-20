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
import cvxpy
import ldpc
import numpy as np
import numpy.typing as npt


def decode(
    matrix: npt.NDArray[np.int_],
    syndrome: npt.NDArray[np.int_],
    *,
    exact: bool = False,
    **decoder_args: object,
) -> npt.NDArray[np.int_]:
    """Find a `vector` that solves `matrix @ vector == syndrome mod 2`.

    If passed an explicit decoder, use it.  If no explicit decoder is provided and `exact is True`,
    solve exactly with an integer linear program.  Otherwise, use a BP-OSD decoder.  In all cases,
    pass the `decoder_args` to the decoder that is used.

    For details about the BD-OSD decoder, see:
    - Documentation: https://roffe.eu/software/ldpc/ldpc/osd_decoder.html
    - Reference: https://arxiv.org/pdf/2005.07016.pdf
    """
    # if a custom decoder was provided, use it
    if callable(custom_decoder := decoder_args.pop("decoder", None)):
        return custom_decoder(matrix, syndrome, **decoder_args)

    if not exact:
        # decode approximate with a BP-OSD decoder
        bposd_decoder = ldpc.bposd_decoder(
            matrix, osd_order=decoder_args.pop("osd_order", 0), **decoder_args
        )
        return bposd_decoder.decode(syndrome)

    # decode exactly by solving an integer linear program
    opt_variables = cvxpy.Variable(matrix.shape[1], boolean=True)
    objective = cvxpy.Minimize(sum(iter(opt_variables)))

    # collect constraints, using slack variables to relax each constraint of the form
    # `expression = val mod 2` to `expression = val + sum_p 2^p s_p`
    constraints = []
    for check, syndrome_bit in zip(matrix, syndrome % 2):
        max_zero = int(sum(check) - syndrome_bit)  # biggest integer that may need to be 0 mod 2
        max_power_of_two = max_zero.bit_length() - 1  # max power of 2 needed for the relaxation
        if max_power_of_two > 0:
            powers_of_two = [2**pp for pp in range(1, max_power_of_two + 1)]
            slack_variables = cvxpy.Variable(max_power_of_two, boolean=True)
            zero_mod_2 = powers_of_two @ slack_variables
        else:
            zero_mod_2 = 0  # pragma: no cover
        constraints.append(check @ opt_variables == syndrome_bit + zero_mod_2)

    # solve the optimization problem!
    problem = cvxpy.Problem(objective, constraints)
    result = problem.solve(**decoder_args)
    if not isinstance(result, float) or not np.isfinite(result):
        message = "Optimal solution to integer linear program could not be found!"
        raise ValueError(message + f"\nSolver output: {result}")
    solution = problem.variables()[0].value
    return np.round(solution).astype(int)
