import math
import random

import numba as nb
import numpy as np


@nb.jit(nopython=True)
def dimension_selector_uniform(n_dimensions):
    return random.randrange(n_dimensions)


def get_dimension_selector_expovariate(
    lambd=None,
    rel_lambd=None,
):
    if lambd is not None and rel_lambd is not None:
        raise ValueError("Cannot set both lambd and rel_lambd")

    if lambd is None and rel_lambd is None:
        # the default, using a rel_lambd of 4.0, placing the pseudo-mean
        # 1/4 of the way into the dimensions
        rel_lambd = 4.

    @nb.jit(nopython=True)
    def dimension_selector_expovariate(n_dimensions):
        nonlocal lambd, rel_lambd

        value = math.inf
        while value >= n_dimensions:
            value = random.expovariate(
                lambd if lambd is not None else rel_lambd/n_dimensions
            )
        return int(value)

    return dimension_selector_expovariate


def get_finder_for_cluster_obeying(
    check_func,
    min_count=1,
    max_count=-1,
    max_depth=-1,
    dimension_selector=dimension_selector_uniform,
    verbose=False,
):
    @nb.jit(nopython=True)
    def _find_cluster_obeying(
        dimensional_parameters,
        non_dimensional_parameters,
        random_seed=None,
        iterations=-1,
    ):
        if dimensional_parameters.shape[-1] != non_dimensional_parameters.shape[-1]:
            raise ValueError(
                "Minor dimension of dimensional_parameters and "
                "non_dimensional_parameters must match"
            )

        final_max_depth = max_depth if max_depth != -1 else (1 + int(
            math.floor(math.log(dimensional_parameters.shape[-1]) / math.log(2))
        ))

        if final_max_depth < 2:
            raise ValueError("max_depth < 2 makes no sense")

        if random_seed is not None:
            random.seed(random_seed)

        bitmap_stack = np.zeros(
            (final_max_depth, dimensional_parameters.shape[-1]),
            dtype=np.bool_,
        )
        bitmap_stack[0,:] = True
        right_branch_stack = np.zeros((final_max_depth,), dtype=np.int8)

        current_level = 1
        iteration = 0

        while True:
            if right_branch_stack[current_level] == 0:
                chosen_dimension = dimension_selector(dimensional_parameters.shape[0])
                chosen_dimension_vals = \
                    dimensional_parameters[chosen_dimension,...][bitmap_stack[current_level-1,:]]

                chosen_split_point = random.uniform(
                    np.min(chosen_dimension_vals),
                    np.max(chosen_dimension_vals),
                )
                bitmap_stack[current_level,:][bitmap_stack[current_level-1,:]] = \
                    (chosen_dimension_vals >= chosen_split_point)
            elif right_branch_stack[current_level] == 1:
                # invert current_level's bitmap, masked by the previous level's
                bitmap_stack[current_level,:][bitmap_stack[current_level-1,:]] = np.logical_not(
                    bitmap_stack[current_level,:][bitmap_stack[current_level-1,:]]
                )
            else:
                # tidy up then unwind
                right_branch_stack[current_level] = 0
                bitmap_stack[current_level,:] = False
                if current_level > 1:
                    current_level-=1
                    # advance branch at underlying level
                    right_branch_stack[current_level]+=1
                else:
                    # we're at the root
                    iteration+=1
                    if iterations != -1 and iteration >= iterations:
                        return None

                    # start again by continuing at the
                    # same current_level

                continue

            bitmap = bitmap_stack[current_level,:]
            remaining_count = np.sum(bitmap)

            if verbose:
                print("current_level = ", current_level, " remaining_count = ", remaining_count)

            if remaining_count < min_count:
                right_branch_stack[current_level]+=1
                continue

            if max_count == -1 or remaining_count <= max_count:
                check_result = check_func(non_dimensional_parameters[...,bitmap])
                if check_result:
                    if check_result > 0:
                        return bitmap
                    else:
                        # negative result signals to stop checking this branch
                        right_branch_stack[current_level]+=1
                        continue

            if remaining_count <= 1:
                # dividing any more makes no sense
                right_branch_stack[current_level]+=1
                continue

            if current_level+1 >= final_max_depth:
                # can't descend any deeper
                right_branch_stack[current_level]+=1
                continue

            current_level+=1

    return _find_cluster_obeying
