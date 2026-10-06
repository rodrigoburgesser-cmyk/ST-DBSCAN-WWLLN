import numpy as np
from datetime import datetime
from math import radians, cos
import csv
from pathlib import Path
from sklearn.neighbors import BallTree


# --------------------------------------------------
# Parameters
# --------------------------------------------------

EPS_KM = 10
EPS_MIN = 5
MIN_SAMPLES = 2

# Input and output directories
INPUT_DIR = Path("data")
OUTPUT_DIR = Path("output")

# Pattern for WWLLN input files
INPUT_PATTERN = "A*.loc"


def process_file(filepath, output_dir):
    """
    Process one WWLLN lightning-location file using ST-DBSCAN.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to the WWLLN input file.
    output_dir : pathlib.Path
        Directory where the output CSV file will be saved.
    """

    # Extract date from filename:
    # Example: A20250121.loc -> 20250121
    filename_raw = filepath.name
    date_part = filename_raw[1:9]

    output_filename = output_dir / f"C{date_part}.csv"

    print(f"Reading {filename_raw} -> Output: {output_filename.name}")

    lat_list = []
    lon_list = []
    time_dt = []

    with filepath.open("r") as f:
        for line in f:
            parts = line.strip().split(",")

            if len(parts) < 4:
                continue

            try:
                dt = datetime.strptime(
                    parts[0] + " " + parts[1],
                    "%Y/%m/%d %H:%M:%S.%f"
                )

                lat_list.append(float(parts[2]))
                lon_list.append(float(parts[3]))
                time_dt.append(dt)

            except (ValueError, TypeError):
                continue

    if not lat_list:
        print(f"No valid data found in {filename_raw}")
        return

    # --------------------------------------------------
    # Sort observations by time
    # --------------------------------------------------

    data_sorted = sorted(
        zip(time_dt, lat_list, lon_list),
        key=lambda x: x[0]
    )

    time_dt, lat, lon = zip(*data_sorted)

    lat = np.array(lat)
    lon = np.array(lon)

    n = len(lat)

    # --------------------------------------------------
    # Convert time to minutes
    # --------------------------------------------------

    t0 = time_dt[0]

    time_min_bin = np.array(
        [
            (t - t0).total_seconds() // 60
            for t in time_dt
        ],
        dtype=int
    )

    # --------------------------------------------------
    # Spatial neighbors using BallTree
    # --------------------------------------------------

    coords_rad = np.radians(
        np.column_stack((lat, lon))
    )

    tree = BallTree(
        coords_rad,
        metric="haversine"
    )

    radius = EPS_KM / 6371.0

    all_spatial_neighbors = tree.query_radius(
        coords_rad,
        r=radius
    )

    # --------------------------------------------------
    # Spatiotemporal neighbors
    # --------------------------------------------------

    st_neighbors_list = []
    core_points = np.zeros(n, dtype=bool)

    for i in range(n):

        s_idx = all_spatial_neighbors[i]

        t_mask = (
            np.abs(
                time_min_bin[s_idx] -
                time_min_bin[i]
            ) <= EPS_MIN
        )

        neighbors = s_idx[t_mask]

        st_neighbors_list.append(neighbors)

        if len(neighbors) >= MIN_SAMPLES:
            core_points[i] = True

    # --------------------------------------------------
    # ST-DBSCAN
    # --------------------------------------------------

    labels = np.full(n, -1)

    visited = np.zeros(n, dtype=bool)

    cluster_id = 0

    for i in range(n):

        if visited[i] or not core_points[i]:
            continue

        labels[i] = cluster_id
        visited[i] = True

        queue = list(st_neighbors_list[i])

        idx = 0

        while idx < len(queue):

            p = queue[idx]

            if labels[p] == -1:
                labels[p] = cluster_id

            if not visited[p] and core_points[p]:

                visited[p] = True

                for nn in st_neighbors_list[p]:

                    if not visited[nn]:
                        queue.append(nn)

            idx += 1

        cluster_id += 1

    # --------------------------------------------------
    # Generate cluster summary
    # --------------------------------------------------

    unique_labels = [
        c for c in np.unique(labels)
        if c != -1
    ]

    summary = []

    for c in unique_labels:

        idxs = np.where(labels == c)[0]

        c_lats = lat[idxs]
        c_lons = lon[idxs]

        c_times = [
            time_dt[i]
            for i in idxs
        ]

        centroid_lat = np.mean(c_lats)

        lat_span = (
            np.max(c_lats) -
            np.min(c_lats)
        ) * 111.0

        lon_span = (
            np.max(c_lons) -
            np.min(c_lons)
        ) * 111.0 * cos(
            radians(centroid_lat)
        )

        summary.append(
            {
                "cluster_id": c,
                "start_time": min(c_times).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "duration_sec": (
                    max(c_times) -
                    min(c_times)
                ).total_seconds(),
                "num_events": len(idxs),
                "centroid_lat": round(
                    centroid_lat,
                    6
                ),
                "centroid_lon": round(
                    np.mean(c_lons),
                    6
                ),
                "area_km2": round(
                    lat_span * lon_span,
                    6
                )
            }
        )

    # --------------------------------------------------
    # Save output
    # --------------------------------------------------

    if summary:

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        with output_filename.open(
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=summary[0].keys()
            )

            writer.writeheader()
            writer.writerows(summary)

        print(
            f"Done: {len(unique_labels)} "
            f"clusters saved to "
            f"{output_filename.name}"
        )


# --------------------------------------------------
# Main execution
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_files = sorted(
        INPUT_DIR.glob(INPUT_PATTERN)
    )

    if not all_files:

        print(
            "No WWLLN input files were found "
            f"in '{INPUT_DIR}'."
        )

        return

    print(
        f"Found {len(all_files)} input files."
    )

    for filepath in all_files:

        process_file(
            filepath,
            OUTPUT_DIR
        )


if __name__ == "__main__":
    main()
