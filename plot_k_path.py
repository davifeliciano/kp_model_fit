from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mpl_config import latex_preamble

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": latex_preamble,
    }
)

csv_dir = Path("data/csv")
plot_dir = Path("plots")

if __name__ == "__main__":

    csv_files = list(csv_dir.glob("*.csv"))
    titles = ("$\ch{CrS2}$", "$\ch{CrSe2}$")

    for file, title in zip(csv_files, titles):
        crystal = file.stem.split("_")[0]

        # Reading data from files
        print(f"For {file}:")
        print("Reading data")
        df = pd.read_csv(file)
        kx = df.loc[:, "kx"].to_numpy()
        ky = df.loc[:, "ky"].to_numpy()

        # Creating plots
        print("Creating path plot")
        fig, ax = plt.subplots()
        ax.set(title=title, ylabel=r"$k_y$", xlabel=r"$k_x$")
        ax.plot(kx, ky, color="blue")
        ax.axis("equal")
        ax.grid()

        # Saving figures
        filename = plot_dir.joinpath(f"{crystal}_k_path.png")
        print(f"Saving figure as {filename}\n")
        plt.savefig(filename, dpi=300)
