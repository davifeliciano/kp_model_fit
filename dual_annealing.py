from __future__ import annotations
from pathlib import Path
import pandas as pd
from numpy.typing import ArrayLike
import numpy as np
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt
from mpl_config import latex_preamble, xtick_label_formatter
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

params_labels = (
    "energy",
    "delta",
    "lambda_c",
    "lambda_v",
    "gamma_0",
    "gamma_1",
    "gamma_2",
    "gamma_3",
    "gamma_4",
    "gamma_5",
    "gamma_6",
)

dual_annealing_contexts = (
    "Minimum detected in the annealing process",
    "Detection occurred in the local search process",
    "Detection done in the dual annealing process",
)


def dual_annealing_callback(x, f, context_index):
    print(f"\nContext: {dual_annealing_contexts[context_index]}")
    print(f"Function value: {f: .4e}")
    print("Params:")
    for param, param_label in zip(iter(x), params_labels):
        print(10 * " ", f"{param_label: <8} = {param: .4e}")


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

        def obj_function(params: ArrayLike) -> float:
            return avg_squared_diff(
                ks=fitting_ks,
                sorted_energies=fitting_energies,
                ham_factory=third_order_ham_factory,
                params=(lattice, *params),
            )

        result = dual_annealing(
            obj_function,
            bounds=np.array(suggested_search_region),
            callback=dual_annealing_callback,
        )
        best_func_value = result.fun
        params = result.x
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
            label="k.p Dual Annealing Fit",
        )

        # Removing repeated entries from legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        filename = plot_dir.joinpath(f"{crystal}_dual_annealing.png")
        print(f"Saving figure as {filename}")
        plt.savefig(filename, dpi=300)

        # Creating dataframe with the results
        print("Creating a dataframe with the results")
        output_df = pd.DataFrame(
            data=[best_func_value, lattice] + list(params),
            index=(
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

        filename = csv_dir.joinpath(f"{crystal}_result_dual_annealing.csv")
        print(f"Saving output as {filename}")
        output_df.to_csv(filename, header=None)
