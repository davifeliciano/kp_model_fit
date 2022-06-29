from pathlib import Path
import re
import pandas as pd

data_dir = Path(__file__).parent
original_dir = data_dir.joinpath("original")
csv_dir = data_dir.joinpath("csv")
original_files = list(original_dir.glob("*.out"))

regex = re.compile(
    r"""
    \s*(\S*?)\s+(\S*?)\s+(\S*?)\s*\n
    \s*(\S*?)\s*\n
    \s*(\S*?)\s*\n
    \s*(\S*?)\s*\n
    \s*(\S*?)\s*\n
    """,
    flags=re.VERBOSE,
)

for file in original_files:
    with file.open() as f:
        print(f"Reading file {file}")
        file_content = f.read()

    print(f"Matching data in {file}")
    rows = []
    for match_ in regex.finditer(file_content):
        rows.append([float(match_.group(i)) for i in range(1, 8)])

    df = pd.DataFrame(rows, columns=("kx", "ky", "kz", "e1", "e2", "e3", "e4"))
    output_file = csv_dir.joinpath(f"{file.stem}.csv")
    print(f"Saving {len(df) - 1} records into file {output_file}")
    df.to_csv(output_file, index=None)
