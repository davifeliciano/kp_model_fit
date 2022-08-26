from __future__ import annotations
import argparse
import logging
import sys
from typing import List
from pathlib import Path
import multiprocessing as mp
import pandas as pd
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
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
GENS = 100
POP_SIZE = 500
REFINEMENT_RATIO = 0.1
THRESHOLD = 1e-5

CRS2_LATTICE = 3.022302679
CRSE2_LATTICE = 3.167287237
CRS2_ENERGY = 0.3536
CRSE2_ENERGY = 0.8903
Y_MARGIN = 0.1

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
            f"of the genetic algorithms. Default is {POP_SIZE}."
        ),
    )

    parser.add_argument(
        "--gens",
        type=int,
        nargs="?",
        default=GENS,
        const=GENS,
        help=(
            "the number of generations to evaluate in the evolution "
            f"process of each population. Default is {GENS}."
        ),
    )

    parser.add_argument(
        "--processes",
        type=int,
        nargs="?",
        default=PROCESSES,
        const=PROCESSES,
        help=(
            "the number of generations to evaluate in the evolution "
            f"process of each population. Default is {PROCESSES}."
        ),
    )

    parser.add_argument(
        "--order",
        type=int,
        choices=(1, 2, 3),
        nargs="?",
        default=3,
        const=3,
        help=(
            "the order of the k vector in the k.p model expansion "
            "for the hamiltonian of the system. Default is 3."
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
        while ga_instance.gen < GENS:
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


if __name__ == "__main__":

    args = parse_args()

    method = args.method
    pop_size = args.pop_size
    gens = args.gens
    processes = args.processes
    order = args.order
    fix = args.fix

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
        ham_factory = ham_factories[order - 1]

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

        # Data subset that will be used in the fitting process
        lower_fit_bound, upper_fit_bound = get_fitting_region(ks)
        fitting_ks = ks[lower_fit_bound:upper_fit_bound, :]
        fitting_energies = sorted_energies[lower_fit_bound:upper_fit_bound, :]
        logger.info(
            "Using interval "
            f"{fitting_ks[0, 0]: .3f} < kx < {fitting_ks[-1, 0]: .3f} "
            "as fitting region"
        )

        # The genetic_algorithm method search for points of
        # maximum, while the dual annealing method seach for
        # points of minimum. This factor flips the objective
        # function according with the method used
        obj_function_factor = -1 if method == "genetic_algorithm" else 1

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
                (expected_energy - THRESHOLD, expected_energy + THRESHOLD),  # energy
                (expected_delta - THRESHOLD, expected_delta + THRESHOLD),  # delta
                (
                    expected_lambda_c - THRESHOLD,
                    expected_lambda_c + THRESHOLD,
                ),  # lambda_c
                (
                    expected_lambda_v - THRESHOLD,
                    expected_lambda_v + THRESHOLD,
                ),  # lambda_v
                (-0.5, 0.5),  # gamma_0
            ]
        else:
            first_order_search_region = [
                (0.0, 1.0),  # energy
                (0.0, 1.2),  # delta
                (0.0, 0.2),  # lambda_c
                (0.0, 0.2),  # lambda_v
                (-0.5, 0.5),  # gamma_0
            ]

        higher_order_search_region = [(-0.5, 0.5) for _ in range(3 * order - 3)]
        suggested_search_region = first_order_search_region + higher_order_search_region

        logger.info("Starting optimization process")
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
                    fit_func_param=10.0,
                )
                for _ in range(processes)
            ]

            logger.info(
                f"Evolving {processes} populations with {pop_size} "
                f"individuals for {gens} generations"
            )

            gas = evolve_gas(gas)

            refined_interval_sizes = tuple(
                [
                    REFINEMENT_RATIO * (upper - lower) / 2
                    for (lower, upper) in suggested_search_region
                ]
            )

            # Evolving gas with refined search regions
            gas = [
                NumericalOptimizationGA(
                    search_region=search_region,
                    function=obj_function,
                    pop_size=pop_size,
                    elite=elite,
                    fit_func_param=10.0,
                )
                for search_region in get_refined_search_regions(gas)
            ]

            logger.info(
                "Restarting process with refined search "
                "regions arround the best individuals"
            )

            gas = evolve_gas(gas)
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
            result = dual_annealing(
                obj_function,
                bounds=np.array(suggested_search_region),
                maxiter=2000,
                no_local_search=True,
            )

            best_func_value = result.fun

            # Data used in the plot
            label = "Dual Annelaing Fit"
            params = result.x
            sorted_eigenvalues = get_energies(
                ks, ham_factory=ham_factory, params=[lattice, *params]
            )

        logger.info(f"Best function value reached: {best_func_value: .3e}")

        # Creating plots
        logger.info("Creating energy plot")
        fig, ax = plt.subplots()

        # Setting up the axes
        ax.set(
            title=title,
            ylabel=r"Energy (\si{\eV})",
            ylim=(
                np.min(sorted_energies[:, 0]) - Y_MARGIN,
                np.max(sorted_energies[:, -1]) + Y_MARGIN,
            ),
        )

        ax.xaxis.set_major_formatter(plt.FuncFormatter(xtick_label_formatter))

        plot_domain = get_plot_domain(ks)
        ax.plot(plot_domain, sorted_energies, color="blue", label="DFT")
        ax.plot(
            plot_domain,
            sorted_eigenvalues,
            color="red",
            label=label,
        )

        # Removing repeated entries from legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        basename = f"{lattice_name}_{method}_order_{order}"
        if fix is True:
            basename += "_fix"

        filename = plot_dir.joinpath(f"{basename}.png")
        logger.info(f"Saving figure as {filename}")
        plt.savefig(filename, dpi=300)

        # Creating dataframe with the results
        logger.info("Creating a dataframe with the results")
        output_df = pd.DataFrame(
            data=np.array(
                [
                    # Fitted values for the params
                    [best_func_value, lattice] + list(params),
                    # Expected values for the params
                    [
                        None,
                        None,
                        expected_energy,
                        expected_delta,
                        expected_lambda_c,
                        expected_lambda_v,
                    ]
                    + [None for _ in range(len(params) - 4)],
                ]
            ).transpose(),
            index=[
                "obj_func_value",
                "lattice",
                "fermi_energy",
                "delta",
                "lamdba_c",
                "lambda_v",
                "gamma_0",
            ]
            + [f"gamma_{n}" for n in range(3 * (order - 1))],
            columns=("fitted_values", "expected_values"),
        )

        filename = results_dir.joinpath(f"{basename}.csv")
        logger.info(f"Saving output as {filename}")
        output_df.to_csv(filename)
