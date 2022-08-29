from __future__ import annotations
import argparse
import logging
import sys
from typing import List, Callable
from pathlib import Path
import multiprocessing as mp
import pandas as pd
from numpy.typing import ArrayLike, NDArray
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
from scipy.optimize import OptimizeResult
from mpl_config import latex_preamble, xtick_label_formatter
from num_opt_ga import NumericalOptimizationGA
from kp_model import (
    first_order_ham_factory,
    second_order_ham_factory,
    third_order_ham_factory,
    get_k_k_index,
    get_energies,
    avg_squared_diff,
    get_fitting_region,
    get_plot_domain,
)

PROCESSES = mp.cpu_count()
THRESHOLD = 1e-5

# Genetic algorithm default properties
GENS = 100
POP_SIZE = 500
MUT_PROB = 0.05
REFINEMENT_RATIO = 0.1

# Dual annealing default properties
ITERS = 1000
INIT_TEMP = 5230

# Physical properties of each lattice
CRS2_LATTICE = 3.022302679
CRSE2_LATTICE = 3.167287237
CRS2_ENERGY = 0.3536
CRSE2_ENERGY = 0.8903

Y_MARGIN = 0.1
LOWER_FIT_LIM = -0.2
UPPER_FIT_LIM = 0.2

# Setting up logger
LOG_FORMAT = "%(levelname)s : %(asctime)s : %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger("logger")
logger.setLevel("INFO")

# Setting up argparse
description = """
    A tool to evaluate the energy bands of the k.p model
    for CrS2 and CrSe2 by means of either a genetic algorithm
    or the dual annealing method.
    """
parser = argparse.ArgumentParser(description=description)

# Setting up Matploltib
plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": latex_preamble,
    }
)


def parse_args():
    """
    Parse the command line arguments using argparse
    """

    parser.add_argument(
        "--processes",
        type=int,
        nargs="?",
        default=PROCESSES,
        const=PROCESSES,
        help=(
            "the number of processes to spawn working on the optimizing "
            f"process. Default is {PROCESSES}, the number of logical "
            "processors in the machine"
        ),
    )

    parser.add_argument(
        "--orders",
        type=int,
        choices=(1, 2, 3),
        nargs="*",
        default=(1, 2, 3),
        help=(
            "the order of the k vector in the k.p model expansion "
            "for the hamiltonian of the system to include in the "
            "energy bands plot. Default is (1, 2, 3)."
        ),
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=("genetic_algorithm", "dual_annealing"),
        nargs="?",
        default="genetic_algorithm",
        const="genetic_algorithm",
        help="the optimizer method to use",
    )

    parser.add_argument(
        "--lattices",
        type=str,
        choices=("crs2", "crse2"),
        nargs="*",
        default=("crs2", "crse2"),
        help="the lattices whose energy bands will be evaluated",
    )

    parser.add_argument(
        "--pop-size",
        type=int,
        nargs="?",
        default=POP_SIZE,
        const=POP_SIZE,
        help=(
            "the number of individuals to compose the populations "
            f"when using genetic algorithms. Default is {POP_SIZE}."
        ),
    )

    parser.add_argument(
        "--gens",
        type=int,
        nargs="?",
        default=GENS,
        const=GENS,
        help=(
            "when using generetic algorithms, the number of generations "
            "to evaluate in the evolution process of each population. "
            f"Default is {GENS}."
        ),
    )

    parser.add_argument(
        "--mut-prob",
        type=float,
        nargs="?",
        default=MUT_PROB,
        const=MUT_PROB,
        help=(
            "when using genetic algorithms, the probability of mutation "
            f"per gene in the recombination process. Default is {MUT_PROB}."
        ),
    )

    parser.add_argument(
        "--refine-ratio",
        type=float,
        nargs="?",
        default=REFINEMENT_RATIO,
        const=REFINEMENT_RATIO,
        help=(
            "when using genetic algorithms, the ratio by which the search region "
            "is shrinked in the second run around the best candidates of minimum. "
            f"If 0.0, no refinement strategy is used. Default is {REFINEMENT_RATIO}."
        ),
    )

    parser.add_argument(
        "--iters",
        type=int,
        nargs="?",
        default=ITERS,
        const=ITERS,
        help=(
            "when using dual annealing method, the max number of iterations "
            f"to perform. Default is {ITERS}."
        ),
    )

    parser.add_argument(
        "--temp",
        type=float,
        nargs="?",
        default=INIT_TEMP,
        const=INIT_TEMP,
        help=(
            "when using dual annealing method, the initial temperature. "
            "use higher values to facilitates a wider search. "
            f"Default is {INIT_TEMP}. Range is (0.01, 5.e4]."
        ),
    )

    parser.add_argument(
        "--no-local-search",
        action="store_true",
        help=(
            "when using dual annealing, perform a traditional generalized "
            "simulated annealing with no local search strategy applied"
        ),
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help=(
            "fix the non-gamma parameters, replacing them by their "
            "respective expected values, infered from the input data"
        ),
    )

    parser.add_argument(
        "--fit-region",
        type=float,
        nargs=2,
        default=(LOWER_FIT_LIM, UPPER_FIT_LIM),
        help=(
            "two floats representing the the region of the data to "
            "consider in the fitting process. Range of each value is "
            "[-1.0, 1.0], -1.0 corresponding to gamma, 0.0, to K, and "
            f"1.0 to M. Default is ({LOWER_FIT_LIM}, {UPPER_FIT_LIM})."
        ),
    )

    return parser.parse_args()


def get_refined_search_regions(
    ga_list: List[NumericalOptimizationGA],
) -> List[NumericalOptimizationGA]:
    return [
        [
            (pos - size, pos + size)
            for pos, size in zip(ga.best()[0].pos, refined_interval_sizes)
        ]
        for ga in ga_list
    ]


def get_best_func_value(ga_instance: NumericalOptimizationGA) -> float:
    return abs(ga_instance.function(ga_instance.best()[0].pos))


def evolve_ga(ga_instance: NumericalOptimizationGA) -> NumericalOptimizationGA | None:
    try:
        while ga_instance.gen < gens:
            ga_instance.evolve()
        return ga_instance
    # Ignore KeyboardInterrupt on a child process
    except KeyboardInterrupt:
        return None


def evolve_gas(
    ga_list: List[NumericalOptimizationGA], processes: int = PROCESSES
) -> List[NumericalOptimizationGA]:
    with mp.Pool(processes) as pool:
        try:
            return pool.map(evolve_ga, ga_list)
        except KeyboardInterrupt:
            # Kill the pool when KeyboardInterrupt is raised
            logger.info("Process terminated by the user")
            pool.terminate()
            pool.join()
            sys.exit(1)


def custom_dual_annealing(
    obj_function: Callable[[NDArray], float],
    bounds: ArrayLike,
    maxiter: int,
    initial_temp: float,
    no_local_search: bool,
) -> OptimizeResult:
    return dual_annealing(
        obj_function,
        bounds=bounds,
        maxiter=maxiter,
        initial_temp=initial_temp,
        no_local_search=no_local_search,
    )


if __name__ == "__main__":

    args = parse_args()

    method = args.method
    pop_size = args.pop_size
    gens = args.gens
    mut_prob = args.mut_prob
    refine_ratio = args.refine_ratio
    iters = args.iters
    temp = args.temp
    no_local_search = args.no_local_search
    processes = args.processes
    orders = args.orders
    fix = args.fix
    lower_fit_region, upper_fit_region = sorted(args.fit_region)

    # Checking if number of processes is not greater than available cores
    if processes > PROCESSES and method == "genetic_algorithm":
        processes = PROCESSES
        logger.warning(
            "Number of processes greater than available logical processors. "
            f"Using {PROCESSES} processes instead."
        )

    csv_dir = Path("data/csv")
    plot_dir = Path("plots")
    results_dir = Path("results")

    lattices = {"crs2": CRS2_LATTICE, "crse2": CRSE2_LATTICE}
    expected_energies = {"crs2": CRS2_ENERGY, "crse2": CRSE2_ENERGY}
    titles = {"crs2": r"$\ch{CrS2}$", "crse2": r"$\ch{CrSe2}$"}
    files = {key: csv_dir.joinpath(f"{key}_data.csv") for key in lattices.keys()}
    ham_factories = (
        first_order_ham_factory,
        second_order_ham_factory,
        third_order_ham_factory,
    )

    for lattice_name in args.lattices:

        lattice = lattices[lattice_name]
        title = titles[lattice_name]
        file = files[lattice_name]

        # Reading data from files
        logger.info(f"Reading data from file {file}")
        df = pd.read_csv(file)
        ks = df.loc[:, "kx":"kz"].to_numpy()
        energies = df.loc[:, "e1":"e4"].to_numpy()
        sorted_energies = np.sort(energies)

        # Expected values of the parameters
        k_k_index = get_k_k_index(ks)
        k_k_energies = sorted_energies[k_k_index, :]
        expected_energy = expected_energies[lattice_name]
        expected_delta = k_k_energies[2] - k_k_energies[1]
        expected_lambda_c = k_k_energies[3] - k_k_energies[2]
        expected_lambda_v = k_k_energies[1] - k_k_energies[0]
        expected_params = (
            expected_energy,
            expected_delta,
            expected_lambda_c,
            expected_lambda_v,
        )

        # Data subset that will be used in the fitting process
        lower_fit_index, upper_fit_index = get_fitting_region(
            ks, lower=lower_fit_region, upper=upper_fit_region
        )

        fitting_ks = ks[lower_fit_index:upper_fit_index, :]
        fitting_energies = sorted_energies[lower_fit_index:upper_fit_index, :]
        logger.info(
            "Using interval "
            f"{fitting_ks[0, 0]: .3f} < kx < {fitting_ks[-1, 0]: .3f} "
            "as fitting region (equivalent to range "
            f"[{lower_fit_region}, {upper_fit_region}] in the plot domain)"
        )

        sorted_eigen_list = []
        params_list = []
        best_func_value_list = []

        for order in orders:

            # The genetic_algorithm method search for points of
            # maximum, while the dual annealing method seach for
            # points of minimum. This factor flips the objective
            # function according with the method used
            obj_function_factor = -1 if method == "genetic_algorithm" else 1
            ham_factory = ham_factories[order - 1]

            # Objective function to optimize
            def obj_function(params: ArrayLike) -> float:
                return obj_function_factor * avg_squared_diff(
                    ks=fitting_ks,
                    sorted_energies=fitting_energies,
                    ham_factory=ham_factory,
                    params=(lattice, *params),
                )

            # Region where to find a minimum
            if fix is True:
                first_order_search_region = [
                    (expected_param - THRESHOLD, expected_param + THRESHOLD)
                    for expected_param in expected_params
                ] + [(-1.0, 1.0)]
            else:
                first_order_search_region = [
                    (-1.0, 1.0),  # energy
                    (0.5, 1.2),  # delta
                    (0.0, 1.0),  # lambda_c
                    (0.0, 1.0),  # lambda_v
                    (-1.0, 1.0),  # gamma_0
                ]

            higher_order_search_region = [(-1.0, 1.0) for _ in range(3 * order - 3)]
            suggested_search_region = (
                first_order_search_region + higher_order_search_region
            )

            logger.info(f"Starting optimization process for order {order}")
            if method == "genetic_algorithm":

                # First two integers in ELITE must be even
                elite = (
                    int(0.025 * pop_size) * 2,
                    int(0.05 * pop_size) * 2,
                    int(0.1 * pop_size),
                )

                # Evolving gas
                gas = [
                    NumericalOptimizationGA(
                        search_region=suggested_search_region,
                        function=obj_function,
                        pop_size=pop_size,
                        elite=elite,
                        mut_probs=(mut_prob, mut_prob),
                    )
                    for _ in range(processes)
                ]

                logger.info(
                    f"Evolving {processes} populations with {pop_size} "
                    f"individuals for {gens} generations"
                )

                gas = evolve_gas(gas, processes=processes)

                # Evolving gas with refined search regions
                if refine_ratio != 0.0:
                    refined_interval_sizes = tuple(
                        [
                            refine_ratio * (upper - lower) / 2
                            for (lower, upper) in suggested_search_region
                        ]
                    )

                    gas = [
                        NumericalOptimizationGA(
                            search_region=search_region,
                            function=obj_function,
                            pop_size=pop_size,
                            elite=elite,
                            mut_probs=(mut_prob, mut_prob),
                        )
                        for search_region in get_refined_search_regions(gas)
                    ]

                    logger.info(
                        "Restarting process with refined search "
                        "regions arround the best individuals"
                    )

                    gas = evolve_gas(gas, processes=processes)

                func_values = list(map(get_best_func_value, gas))
                best_func_value = min(func_values)
                best_index = func_values.index(best_func_value)
                ga = gas[best_index]

                # Data used in the plot
                label = "GA Fit"
                params = ga.best()[0].pos
                sorted_eigenvalues = get_energies(
                    ks, ham_factory=ham_factory, params=(lattice, *params)
                )

            if method == "dual_annealing":

                logger.info(
                    f"Spawning {processes} Dual Annealing with "
                    f"temp={temp}, maxiter={iters} and "
                    f"no_local_search={no_local_search}"
                )

                dual_annealing_args = (
                    obj_function,
                    np.array(suggested_search_region),
                    iters,
                    temp,
                    no_local_search,
                )

                with mp.Pool(processes) as pool:
                    promises = [
                        pool.apply_async(custom_dual_annealing, dual_annealing_args)
                        for _ in range(processes)
                    ]
                    results = [promise.get() for promise in promises]

                # Data used in the plot
                label = "Dual Annelaing Fit"
                func_values = [result.fun for result in results]
                best_func_value = min(func_values)
                best_index = func_values.index(best_func_value)
                params = results[best_index].x
                sorted_eigenvalues = get_energies(
                    ks, ham_factory=ham_factory, params=[lattice, *params]
                )

            logger.info(f"Best function value reached: {best_func_value: .3e}")
            sorted_eigen_list.append(sorted_eigenvalues)
            params_list.append(params)
            best_func_value_list.append(best_func_value)

        # Creating plots
        logger.info("Creating energy plot")
        fig, ax = plt.subplots()

        # Setting up the axes
        ylim = (
            np.min(sorted_energies[:, 0]) - Y_MARGIN,
            np.max(sorted_energies[:, -1]) + Y_MARGIN,
        )

        ax.set(
            title=title,
            ylabel=r"Energy (\si{\eV})",
            ylim=ylim,
        )

        ax.xaxis.set_major_formatter(plt.FuncFormatter(xtick_label_formatter))

        plot_domain = get_plot_domain(ks)
        ax.vlines(
            (plot_domain[lower_fit_index], plot_domain[upper_fit_index]),
            *ylim,
            color="black",
            alpha=0.8,
            linestyles="dashed",
            linewidths=0.7,
            label="Fitting Region",
            zorder=-2,
        )

        ax.plot(
            plot_domain,
            sorted_energies,
            color="black",
            marker="o",
            linestyle="none",
            markersize=1.0,
            label="DFT",
        )

        colors = ("blue", "red", "green")

        for sorted_eigenvalues, color, order in zip(sorted_eigen_list, colors, orders):
            ax.plot(
                plot_domain,
                sorted_eigenvalues,
                color=color,
                alpha=0.8,
                label=label + f" (Order {order})",
                zorder=-1,
            )

        # Removing repeated entries from legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), framealpha=1.0)

        # Basename to save the image and the csv file
        joined_orders = "".join([str(order) for order in orders])
        basename = f"{lattice_name}_{method}_order_{joined_orders}"
        if fix is True:
            basename += "_fix"

        filename = plot_dir.joinpath(f"{basename}.png")
        logger.info(f"Saving figure as {filename}")
        plt.savefig(filename, dpi=300)

        # Creating dataframe with the results
        logger.info("Creating a dataframe with the results")

        # The dataframe will have 7 + 3 * max(orders) - 3 = 4 + 3 * max(orders) rows
        rows = 4 + 3 * max(orders)

        fitted_params_columns = [
            [best_func_value, lattice]
            + list(params)
            + [None for _ in range(rows - len(params) - 2)]
            for params, best_func_value in zip(params_list, best_func_value_list)
        ]

        expected_params_column = [None for _ in range(rows)]
        expected_params_column[2:6] = expected_params
        data_list = [
            *fitted_params_columns,
            expected_params_column,
        ]  # Needs to be transposed

        first_order_index_labels = [
            "obj_func_value",
            "lattice",
            "fermi_energy",
            "delta",
            "lamdba_c",
            "lambda_v",
            "gamma_0",
        ]

        higher_order_index_labels = [f"gamma_{n}" for n in range(3 * max(orders) - 3)]

        output_df = pd.DataFrame(
            data=np.array(data_list).transpose(),
            index=first_order_index_labels + higher_order_index_labels,
            columns=[f"order_{order}" for order in orders] + ["expected_values"],
        )

        filename = results_dir.joinpath(f"{basename}.csv")
        logger.info(f"Saving output as {filename}")
        output_df.to_csv(filename)
