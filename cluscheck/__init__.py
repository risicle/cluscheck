import math
import random

import numba as nb
import numpy as np


def get_finder_for_cluster_obeying(
    check_func,
    min_count=1,
    max_count=-1,
    max_depth=-1,
    debug=False,
):
    @nb.jit(nopython=True)
    def _find_cluster_obeying(
        dimensional_parameters,
        non_dimensional_parameters,
        random_seed=0,
    ):
        nonlocal check_func, min_count, max_count, max_depth

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

        random.seed(random_seed)

        bitmap_stack = np.zeros(
            (final_max_depth, dimensional_parameters.shape[-1]),
            dtype=np.bool_,
        )
        bitmap_stack[0,:] = True
        right_branch_stack = np.zeros((final_max_depth,), dtype=np.int8)

        current_level = 1

        while True:
            if right_branch_stack[current_level] == 0:
                chosen_dimension = random.randrange(dimensional_parameters.shape[0])
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
                # else we're at the root, start again by continuing at the
                # same current_level
                continue

            bitmap = bitmap_stack[current_level,:]
            remaining_count = np.sum(bitmap)

            if debug:
                print("current_level = ", current_level, " remaining_count = ", remaining_count)

            if remaining_count < min_count:
                right_branch_stack[current_level]+=1
                continue

            if (
                (max_count == -1 or remaining_count <= max_count)
                and check_func(non_dimensional_parameters[...,bitmap])
            ):
                return bitmap

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
