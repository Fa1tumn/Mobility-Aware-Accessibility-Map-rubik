"""Record IMU samples and wheelchair release/stop pairs for one road surface."""
import csv

from field_experiment import run


def mean_ax(times, ax, start, end):
    """Return the time-weighted mean acceleration."""
    import numpy as np

    if end <= start or len(times) < 2 or start < times[0] or end > times[-1]:
        return None
    first = max(0, times.searchsorted(start, side="right") - 1)
    last = times.searchsorted(end, side="left") + 1
    local = times[first:last]
    if np.any(np.diff(local) > 0.1):
        return None
    interior = local[(local > start) & (local < end)]
    t = np.r_[start, interior, end]
    a = np.interp(t, local, ax[first:last])
    return float(np.sum((a[:-1] + a[1:]) * 0.5 * np.diff(t)) / (end - start))


def summarize_coasts(folder):
    import numpy as np

    with (folder / "imu.csv").open(newline="") as handle:
        data = np.array([(float(row["t_s"]), float(row["ax"])) for row in csv.DictReader(handle)])
    monotonic = bool(len(data)) and np.all(np.diff(data[:, 0]) > 0)
    with (folder / "trials.csv").open(newline="") as handle:
        trials = list(csv.DictReader(handle))
    temporary = folder / "coast_summary.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial", "release_s", "stop_s", "coast_s", "mean_decel_m_s2", "status"])
        for row in trials:
            release, stop = float(row["release_s"]), float(row["stop_s"])
            decel = None
            if monotonic:
                moving = mean_ax(data[:, 0], data[:, 1], release, stop)
                # Use the stationary period after the stop.
                rest = mean_ax(data[:, 0], data[:, 1], stop + 0.5, stop + 2.5)
                if moving is not None and rest is not None:
                    decel = rest - moving
            writer.writerow([row["trial"], release, stop, row["coast_s"], decel,
                             "ok" if decel is not None else "insufficient_imu_coverage"])
    temporary.replace(folder / "coast_summary.csv")


if __name__ == "__main__":
    run("coast", __doc__, summarize=summarize_coasts)
