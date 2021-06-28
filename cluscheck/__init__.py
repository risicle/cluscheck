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
    fixed_dimensional_parameters=-1,
    fixed_non_dimensional_parameters=-1,
    fixed_n=-1,
    verbose=False,
    jit_kwargs={},
):
    @nb.jit(nopython=True, **jit_kwargs)
    def _find_cluster_obeying(
        dimensional_parameters,
        non_dimensional_parameters,
        random_seed=None,
        iterations=-1,
    ):
        if dimensional_parameters.shape[1] != non_dimensional_parameters.shape[0]:
            raise ValueError(
                "Minor dimension of dimensional_parameters must match "
                "major dimension of non_dimensional_parameters"
            )

        if (
            fixed_dimensional_parameters != -1
            and fixed_dimensional_parameters != dimensional_parameters.shape[0]
        ):
            raise ValueError("Number of dimensional parameters not expected value")

        if (
            fixed_non_dimensional_parameters != -1
            and fixed_non_dimensional_parameters != non_dimensional_parameters.shape[1]
        ):
            raise ValueError("Number of non-dimensional parameters not expected value")

        if (
            fixed_n != -1
            and fixed_n != non_dimensional_parameters.shape[0]
        ):
            raise ValueError("Number of candidates not expected value")

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
                vals_count = 0
                # initialize these to any value of the correct type
                vals_min = vals_max = dimensional_parameters[0,0]
                # scan for range of remaining values in this dimension
                for i in range(dimensional_parameters.shape[1]):
                    if bitmap_stack[current_level-1,i]:
                        v = dimensional_parameters[chosen_dimension,i]

                        if vals_count == 0:
                            vals_min = vals_max = v
                        else:
                            vals_min = min(vals_min, v)
                            vals_max = max(vals_max, v)

                        vals_count+=1

                chosen_split_point = random.uniform(vals_min, vals_max)

                # mark values greater than threshold
                remaining_count = 0
                for i in range(dimensional_parameters.shape[1]):
                    if bitmap_stack[current_level-1,i]:
                        is_chosen = (
                            dimensional_parameters[chosen_dimension,i] >= chosen_split_point
                        )
                        bitmap_stack[current_level,i] = is_chosen
                        if is_chosen:
                            remaining_count+=1
            elif right_branch_stack[current_level] == 1:
                # invert current_level's bitmap, masked by the previous level's
                remaining_count = 0
                for i in range(bitmap_stack.shape[1]):
                    if bitmap_stack[current_level-1,i]:
                        is_chosen = not bitmap_stack[current_level,i]
                        bitmap_stack[current_level,i] = is_chosen
                        if is_chosen:
                            remaining_count+=1
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

            if verbose:
                print("current_level = ", current_level, " remaining_count = ", remaining_count)

            if remaining_count < min_count:
                right_branch_stack[current_level]+=1
                continue

            if max_count == -1 or remaining_count <= max_count:
                ndp_subset = np.empty(
                    (remaining_count, non_dimensional_parameters.shape[1],),
                    dtype=non_dimensional_parameters.dtype,
                )

                j = 0
                for i in range(bitmap_stack.shape[1]):
                    if j < remaining_count and bitmap_stack[current_level,i]:
                        for k in range(ndp_subset.shape[1]):
                            ndp_subset[j,k] = non_dimensional_parameters[i,k]
                        j+=1

                check_result = check_func(ndp_subset)
                if check_result:
                    if check_result > 0:
                        return bitmap_stack[current_level,:]
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
