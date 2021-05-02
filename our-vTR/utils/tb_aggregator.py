# tb_aggregator.py

from argparse import ArgumentParser
from glob import glob
from os.path import isdir
from shutil import rmtree
from typing import List

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
from torch.utils.tensorboard import SummaryWriter


def read_tb_events(in_dirs_glob: str) -> dict:
    """Read the TensorBoard event files matching the provided glob pattern
    and return their scalar data as a dict with data tags as keys.

    Args:
        in_dirs_glob (str): Glob pattern of the run directories to read from disk.

    Returns:
        dict: A dictionary of containing scalar run data with keys like
            'train/loss', 'train/mae', 'val/loss', etc.
    """

    summary_iterators = [
        EventAccumulator(dirname).Reload() for dirname in glob(in_dirs_glob)
    ]

    tags = summary_iterators[0].Tags()["scalars"]

    for iterator in summary_iterators:
        # assert all runs have the same tags for scalar data
        assert iterator.Tags()["scalars"] == tags

    out = {t: [] for t in tags}

    for tag in tags:
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len({e.step for e in events}) == 1

            out[tag].append([e.value for e in events])

    return out


def write_reduced_tb_events(
    in_dirs_glob: str, outdir: str, overwrite: bool = False
) -> List[np.ndarray]:
    """Averages TensorBoard runs matching the in_dirs_glob and writes
    the result to outdir.

    Inspired by https://stackoverflow.com/a/48774926.

    Args:
        in_dirs_glob (str): Glob pattern of the run directories to reduce.
        outdir (str): Name of the directory to save the new reduced run data. Will
            have '-mean'/'mean-pm-str' appended for the mean and std.dev. reductions.
        overwrite (bool): Whether to overwrite existing reduction directories.
            Defaults to False.
    Returns:
        np.ndarray: The mean of runs matching in_dirs_glob for each
            recorded scalar and at each time step.
    """

    if isdir(outdir):
        if not overwrite:
            raise FileExistsError(
                f"'{outdir}' already exists, pass overwrite=True to proceed anyway"
            )

        try:
            rmtree(outdir)
        except FileNotFoundError:  # [Errno 2] No such file or directory
            pass

    n_runs = len(glob(in_dirs_glob))
    assert n_runs > 0, f"No runs found for glob pattern '{in_dirs_glob}'"

    print(f"\nReducing {len(glob(in_dirs_glob))} TensorBoard runs to '{outdir}'\n")

    runs_to_reduce = read_tb_events(in_dirs_glob)

    tags, values = zip(*runs_to_reduce.items())

    # values = np.array(values)

    # write mean reduction
    writer = SummaryWriter(outdir)

    # timestep_mean = values.mean(axis=-1)

    timestep_mean = [np.array(metric).mean(axis=-1) for metric in values]

    for tag, means in zip(tags, timestep_mean):
        for idx, mean in enumerate(means):
            writer.add_scalar(tag, mean, idx)

    # Important for allowing reduce_tb_events to overwrite. Without it,
    # try_rmtree will raise OSError: [Errno 16] Device or resource busy
    # trying to delete the existing outdir.
    writer.close()

    return timestep_mean


if __name__ == "__main__":
    parser = ArgumentParser("TensorBoard Run Reducer")

    parser.add_argument(
        "-d",
        "--in-dirs-glob",
        help=(
            "Glob pattern of the run directories to reduce. "
            "Remember to protect wildcards with quotes to prevent shell expansion."
        ),
    )
    parser.add_argument(
        "-o", "--outdir", help="Name of the directory to save the new reduced run data."
    )
    parser.add_argument(
        "-f",
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing reduction directories.",
    )

    args, _ = parser.parse_known_args()

    write_reduced_tb_events(**vars(args))
