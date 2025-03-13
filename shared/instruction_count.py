"""
This script will go through all the `kernel-*.traceg` files in the given directory,
and calculate the total number of instructions executed that match the given pattern, stratified by the name of the kernel.

Usage: ./instruction_count.py <directory> <pattern>

Example: ./instruction_count.py ./traces "LDG"
This will calculate the total number of instructions executed that match the pattern "LDG" in all the `kernel-*.traceg` files in the directory `/path/to/directory/`

The output will be in the form:
| <pattern> count | kernel_name |
|-----------------|-------------|
| 123456          | kernel_name |
| 123456          | kernel_name |
...

the output will be in arbitrary order
"""

import os
import sys
from collections import defaultdict
import pandas as pd


def count_instructions(directory: str, pattern: str) -> dict[str, float]:
    # first is the aggregate count of instructions for each kernel,
    # second is the number of times we've seen that kernel
    instruction_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        if filename.startswith("kernel-") and filename.endswith(".traceg"):
            print(f"Processing {filename}")
            file_path: str = os.path.join(directory, filename)

            # the kernel name is in the first line of the file,
            # all trace files start with the same basic structure
            with open(file_path, "r") as file:
                lines = file.readlines().__iter__()
                # the first line is of the form `-kernel name = <kernel_name>`
                kernel_name = next(lines).strip().split()[-1]

                # increment the count for this kernel
                instruction_counts[kernel_name][1] += 1

                # now we skip to the start of the actual instruction traces

                while ((line := next(lines, None)) is not None) and not line.startswith(
                    "insts = "
                ):
                    ...

                # now we are at the start of the instruction traces

                # traces format = [line_num] PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width [adrrescompress?] [mem_addresses] [immediate]

                while (line := next(lines, None)) is not None:
                    if pattern in line:
                        # if matcher.match(line):
                        instruction_counts[kernel_name][0] += 1

    # turn our instruction counts into average counts
    average_counts = {
        kernel_name: count[0] / count[1]
        for kernel_name, count in instruction_counts.items()
    }

    return average_counts


KERNEL_COLUMN_NAME = "kernel_name"
COUNT_COLUMN_NAME = "average count"


def print_instruction_counts(instruction_counts: dict[str, float]):
    # Convert the dictionary to a DataFrame for better formatting
    df = pd.DataFrame(
        list(instruction_counts.items()),
        columns=[KERNEL_COLUMN_NAME, COUNT_COLUMN_NAME],
    )

    # Sort by count in descending order
    df = df.sort_values(by=COUNT_COLUMN_NAME, ascending=False)

    # reorder so that the count is first
    df = df[[COUNT_COLUMN_NAME, KERNEL_COLUMN_NAME]]

    # Print the DataFrame
    print(df.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ./instruction_count.py <directory> <pattern>")
        sys.exit(1)

    directory = sys.argv[1]
    pattern = sys.argv[2]

    instruction_counts = count_instructions(directory, pattern)
    print_instruction_counts(instruction_counts)
