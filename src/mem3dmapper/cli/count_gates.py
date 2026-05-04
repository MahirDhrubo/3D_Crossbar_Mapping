import argparse
from pathlib import Path
import sys
import csv

from mem3dmapper.netlist.parser import parse_netlist

# netlist = parse_netlist(f"data/netlists/adder_nor_netlist/{file_name}.blif")


def parse_directory(dirpath: Path, ext: str = ".blif"):
    if not dirpath.exists():
        print(f"Directory '{dirpath}' does not exist.")
        return 2
    if not dirpath.is_dir():
        print(f"Path '{dirpath}' is not a directory.")
        return 2

    # blif_files = list(dirpath.glob(f"*{ext}"))
    files = sorted(file for file in dirpath.iterdir() if file.is_file() and file.suffix == ext)
    if not files:
        print(f"No '{ext}' files found in directory '{dirpath}'.")
        return 2

    rows: list[int] = []
    for f in files:
        netlist = parse_netlist(str(f))
        # print(f"{f.name}: {len(netlist.gates)} gates")
        rows.append(len(netlist.gates))

        # out_csv = dirpath / "gate_counts.csv"
        # with out_csv.open("a", newline="") as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow([f.name, len(netlist.gates)])

    return rows


def main():
    parser = argparse.ArgumentParser(description="Count gates in BLIF files.")
    parser.add_argument("directory1", type=Path, help="Directory containing BLIF files.")
    parser.add_argument("directory2", type=Path, help="Directory containing BLIF files.")
    parser.add_argument("--ext", type=str, default=".blif", help="File extension to look for (default: .blif)")
    args = parser.parse_args()
    return parse_directory(dirpath = args.directory1, ext = args.ext), parse_directory(dirpath = args.directory2, ext = args.ext)

if __name__ == "__main__":
    row1, row2 = main()

    for i in range(len(row1)):
        print((row2[i] - row1[i]) / row2[i])
