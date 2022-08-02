from __future__ import annotations
import sys
from typing import List
from pathlib import Path
import multiprocessing as mp
import pandas as pd
from numpy.typing import ArrayLike, NDArray
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_config import latex_preamble, xtick_label_formatter
from num_opt_ga import NumericalOptimizationGA
from kp_model import (
    third_order_ham_factory,
    get_energies,
    avg_squared_diff,
    get_fitting_region,
    get_plot_domain,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": latex_preamble,
    }
)

PROCESSES = mp.cpu_count() - 2
GENS = 50
POP_SIZE = 500
# THRESHOLD = 2e-4
REFINEMENT_RATIO = 0.1
CRS2_LATTICE = 3.022302679
CRSE2_LATTICE = 3.167287237

csv_dir = Path("data/csv")
plot_dir = Path("plots")

suggested_search_region = (
    (-1.0, 1.0),  # energy
    (0.0, 2.0),  # delta
    (-0.2, 0.2),  # lambda_c
    (-0.2, 0.2),  # lambda_v
    (-0.5, 0.5),  # gamma_0
    (-0.3, 0.3),  # gamma_1
    (-0.3, 0.3),  # gamma_2
    (-0.3, 0.3),  # gamma_3
    (-0.3, 0.3),  # gamma_4
    (-0.3, 0.3),  # gamma_5
    (-0.3, 0.3),  # gamma_6
)

refined_interval_sizes = tuple(
    [
        REFINEMENT_RATIO * (upper - lower) / 2
        for (lower, upper) in suggested_search_region
    ]
)


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
        while (
            ga_instance.gen < GENS
        ):  # and get_best_func_value(ga_instance) > THRESHOLD:
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
            print("Process terminated by the user")
            pool.terminate()
            pool.join()
            sys.exit(1)


if __name__ == "__main__":

    csv_files = list(csv_dir.glob("*.csv"))
    lattices = (CRS2_LATTICE, CRSE2_LATTICE)
    titles = ("$\ch{CrS2}$", "$\ch{CrSe2}$")

    for file, lattice, title in zip(csv_files, lattices, titles):
        crystal = file.stem.split("_")[0]

        # Reading data from files
        print(f"\nFor {file}:", end="\n\n")
        print("Reading data")
        df = pd.read_csv(file)
        ks = df.loc[:, "kx":"kz"].to_numpy()
        energies = df.loc[:, "e1":"e4"].to_numpy()
        sorted_energies = np.sort(energies)

        # Data subset that will be used in the fitting process
        lower_fit_bound, upper_fit_bound = get_fitting_region(ks)
        fitting_ks = ks[lower_fit_bound:upper_fit_bound, :]
        fitting_energies = sorted_energies[lower_fit_bound:upper_fit_bound, :]
        print(
            f"Fitting region: {fitting_ks[0, 0]: .3f} < kx < {fitting_ks[-1, 0]: .3f}"
        )

        # Objective function to optimize
        def obj_function(params: ArrayLike) -> float:
            return -avg_squared_diff(
                ks=fitting_ks,
                sorted_energies=fitting_energies,
                ham_factory=third_order_ham_factory,
                params=(lattice, *params),
            )

        # Evolving gas
        gas = [
            NumericalOptimizationGA(
                search_region=suggested_search_region,
                function=obj_function,
                pop_size=POP_SIZE,
                elite=(10, 120, 120),
                fit_func_param=10.0,
            )
            for _ in range(PROCESSES)
        ]

        print(f"Evolving initial {PROCESSES} populations")
        gas = evolve_gas(gas)

        # Evolving gas with refined search regions
        gas = [
            NumericalOptimizationGA(
                search_region=search_region,
                function=obj_function,
                pop_size=POP_SIZE,
                elite=(10, 120, 120),
                fit_func_param=10.0,
            )
            for search_region in get_refined_search_regions(gas)
        ]

        print(f"Evolving {PROCESSES} populations with refined search regions")
        gas = evolve_gas(gas)
        func_values = list(map(get_best_func_value, gas))
        best_func_value = min(func_values)
        print(f"Best function value: {best_func_value: .3e}")

        # Selecting relevant data from the ga containing the best individual
        best_index = func_values.index(best_func_value)
        ga = gas[best_index]
        best_gen = ga.gen
        params = ga.best()[0].pos
        sorted_eigenvalues = get_energies(
            ks, ham_factory=third_order_ham_factory, params=[lattice, *params]
        )

        # Creating plots
        print("\nCreating energy plot")
        fig, ax = plt.subplots()
        ax.set(ylabel=r"Energy (\si{\eV})", title=title)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(xtick_label_formatter))

        plot_domain = get_plot_domain(ks)
        ax.plot(plot_domain, sorted_energies, color="blue", label="DFT")
        ax.plot(
            plot_domain,
            sorted_eigenvalues,
            color="red",
            label="k.p GA Fit",
        )

        # Removing repeated entries from legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        filename = plot_dir.joinpath(f"{crystal}.png")
        print(f"Saving figure as {filename}")
        plt.savefig(filename, dpi=300)

        # Creating dataframe with the results
        print("Creating a dataframe with the results")
        output_df = pd.DataFrame(
            data=[ga.gen, best_func_value, lattice] + list(params),
            index=(
                "gen",
                "obj_func_value",
                "lattice",
                "fermi_energy",
                "delta",
                "lamdba_c",
                "lambda_v",
                "gamma_0",
                "gamma_1",
                "gamma_2",
                "gamma_3",
                "gamma_4",
                "gamma_5",
                "gamma_6",
            ),
        )

        filename = csv_dir.joinpath(f"{crystal}_result.csv")
        print(f"Saving output as {filename}")
        output_df.to_csv(filename, header=None)
