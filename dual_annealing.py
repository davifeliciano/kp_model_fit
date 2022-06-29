from __future__ import annotations
from pathlib import Path
import pandas as pd
from numpy.typing import ArrayLike
import numpy as np
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt
from kp_model import (
    third_order_ham_factory,
    get_energies,
    avg_squared_diff,
)

CRS2_LATTICE = 3.022302679
CRSE2_LATTICE = 3.167287237

csv_dir = Path("data/csv")
csv_files = list(csv_dir.glob("*.csv"))
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
    print(f"Function value: {f}")
    print("Params:")
    for param, param_label in zip(iter(x), params_labels):
        print(10 * " ", f"{param_label: <8} = {param}")


if __name__ == "__main__":
    lattices = (CRS2_LATTICE, CRSE2_LATTICE)
    titles = ("$CrS_2$", "$CrSe_2$")
    output_data = []

    for file, lattice, title in zip(csv_files, lattices, titles):
        crystal = file.stem.split("_")[0]

        print(f"\nFor {file}:", end="\n\n")
        print("Reading data")
        df = pd.read_csv(file)
        ks = df.loc[:, "kx":"kz"].to_numpy()
        energies = df.loc[:, "e1":"e4"].to_numpy()
        sorted_energies = np.sort(energies)

        k_fitting_region = (20, 50)
        zero_neigh_ks = ks[k_fitting_region[0] : k_fitting_region[1], :]
        zero_neigh_energies = sorted_energies[
            k_fitting_region[0] : k_fitting_region[1], :
        ]

        def obj_function(params: ArrayLike) -> float:
            return avg_squared_diff(
                ks=zero_neigh_ks,
                sorted_energies=zero_neigh_energies,
                ham_factory=third_order_ham_factory,
                params=(lattice, *params),
            )

        print(f"Fitting for kx in [{zero_neigh_ks[0, 0]}, {zero_neigh_ks[-1, 0]}]")
        result = dual_annealing(
            obj_function,
            bounds=np.array(suggested_search_region),
            callback=dual_annealing_callback,
        )

        # Energy diagrams
        print("\nCreating energy plot")
        fig, ax = plt.subplots()
        ax.set(xlabel="$k_x$", ylabel="Energy", title=title)
        ax.plot(ks[:, 0], sorted_energies, color="blue", label="DFT")

        params = result.x
        best_func_value = result.fun

        # Appending results to output_data to create an unified dataframe later
        output_data.append([best_func_value, lattice] + list(params))

        sorted_eigenvalues = get_energies(
            ks, ham_factory=third_order_ham_factory, params=[lattice, *params]
        )

        ax.plot(
            ks[:, 0],
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
        data=np.array(output_data).transpose(),
        columns=("crs2", "crse2"),
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

    filename = csv_dir.joinpath("result_dual_annealing.csv")
    print(f"Saving output as {filename}")
    output_df.to_csv(filename)
