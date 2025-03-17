# Description: This script contains functions to convert CSV files to HDF5 and .npy files in parallel.
import os

os.environ["KMP_WARNINGS"] = "off"
import csv
import numpy as np
import h5py
from numba import njit
from dask import delayed, compute

# Define structured dtypes for memory efficiency
event_dtype = np.dtype(
    [
        ("evt_id", np.int32),
        ("evt_weight", np.float32),
        ("met", np.float32),
        ("met_phi", np.float32),
        ("num_jets", np.int16),
    ]
)

jet_dtype = np.dtype(
    [
        ("evt_id", np.int32),
        ("jet_index", np.int16),
        ("num_constits", np.int16),
        ("b_tagged", np.int8),
        ("jet_pt", np.float32),
        ("jet_eta", np.float32),
        ("jet_phi", np.float32),
    ]
)

constituent_dtype = np.dtype(
    [
        ("evt_id", np.int32),
        ("jet_index", np.int16),
        ("constituent_index", np.int16),
        ("particle_id", np.int32),
        ("pt", np.float32),
        ("eta", np.float32),
        ("phi", np.float32),
    ]
)


@njit
def calculate_jet_properties_numba(pt_arr, eta_arr, phi_arr):
    """
    Numba-accelerated calculation of jet properties using vectorized NumPy operations.
    Processes all constituents in a jet simultaneously.

    """
    px = 0.0
    py = 0.0
    pz = 0.0
    for i in range(pt_arr.shape[0]):
        pt = pt_arr[i]
        eta = eta_arr[i]
        phi = phi_arr[i]
        px += pt * np.cos(phi)
        py += pt * np.sin(phi)
        pz += pt * np.sinh(eta)
    jet_pt = np.sqrt(px * px + py * py)
    jet_phi = np.arctan2(py, px)
    jet_eta = np.arcsinh(pz / jet_pt) if jet_pt != 0.0 else 0.0
    return jet_pt, jet_eta, jet_phi


def process_event(evt_id, row):
    """
    Process a single event row:
      - Convert string data to floats.
      - Extract event-level data.
      - Loop over jets and their constituents.
      - Use vectorized (Numba) calculation for jet properties.

    """
    try:
        row_floats = [float(x) for x in row]
    except ValueError:
        return None  # Skip events with conversion issues.

    row_np = np.array(row_floats, dtype=np.float32)
    evt_weight = row_np[0]
    met = row_np[1]
    met_phi = row_np[2]
    num_jets = int(row_np[3])
    event_data = [evt_id, evt_weight, met, met_phi, num_jets]

    jet_offset = 4  # First 4 columns are event-level variables.
    jets = []
    constituents = []

    for i in range(num_jets):
        num_constits = int(row_np[jet_offset])
        b_tagged = int(row_np[jet_offset + 1])
        jet_constituents = []
        for j in range(num_constits):
            base = jet_offset + 2 + j * 4
            pid = row_np[base]
            pt = row_np[base + 1]
            eta = row_np[base + 2]
            phi = row_np[base + 3]
            jet_constituents.append((pt, eta, phi))
            constituents.append((evt_id, i, j, pid, pt, eta, phi))
        if jet_constituents:
            arr = np.array(jet_constituents, dtype=np.float32)
            pt_arr = arr[:, 0]
            eta_arr = arr[:, 1]
            phi_arr = arr[:, 2]
            jet_pt, jet_eta, jet_phi = calculate_jet_properties_numba(
                pt_arr, eta_arr, phi_arr
            )
        else:
            jet_pt, jet_eta, jet_phi = 0.0, 0.0, 0.0
        jets.append((evt_id, i, num_constits, b_tagged, jet_pt, jet_eta, jet_phi))
        jet_offset += 2 + num_constits * 4

    return event_data, jets, constituents


def csv_chunk_generator(csv_file, chunk_size=10000):
    """
    Generator yielding CSV file chunks (lists of rows) to avoid loading the entire file.
    Skips empty rows and rows with 'evtwt' in the first column.

    """
    with open(csv_file, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header row.
        chunk = []
        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue
            if "evtwt" in row[0].lower():
                continue
            chunk.append(tuple(row))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def append_to_hdf5(h5file, dataset_name, data_chunk):
    """
    Append a data chunk to an existing resizable HDF5 dataset.

    """
    ds = h5file[dataset_name]
    current_rows = ds.shape[0]
    num_cols = ds.shape[1]
    new_rows = current_rows + data_chunk.shape[0]
    ds.resize((new_rows, num_cols))
    ds[current_rows:new_rows, :] = data_chunk


def process_chunk(chunk, start_evt_id):
    """
    Process a chunk of CSV rows into homogeneous 2D arrays.
    Instead of creating structured arrays (with named fields), we build
    plain lists of lists and convert them to a homogeneous NumPy array.

    """
    events = []  # Each event: [evt_id, evt_weight, met, met_phi, num_jets]
    jets = (
        []
    )  # Each jet: [evt_id, jet_index, num_constits, b_tagged, jet_pt, jet_eta, jet_phi]
    constituents = (
        []
    )  # Each constituent: [evt_id, jet_index, constituent_index, particle_id, pt, eta, phi]
    evt_id = start_evt_id
    for row in chunk:
        res = process_event(evt_id, row)
        if res is not None:
            ev, js, cons = res
            events.append(ev)  # ev is already a list of numbers
            jets.extend(js)  # each element in js is a list of numbers
            constituents.extend(cons)  # same here
        evt_id += 1
    # Create homogeneous arrays. All columns will be stored as float32.
    events_arr = (
        np.array(events, dtype=np.float32)
        if events
        else np.empty((0, 5), dtype=np.float32)
    )
    jets_arr = (
        np.array(jets, dtype=np.float32) if jets else np.empty((0, 7), dtype=np.float32)
    )
    constituents_arr = (
        np.array(constituents, dtype=np.float32)
        if constituents
        else np.empty((0, 7), dtype=np.float32)
    )
    return events_arr, jets_arr, constituents_arr


def convert_csv_to_hdf5_npy_parallel(
    csv_file,
    output_prefix,
    out_path,
    file_type="h5",
    chunk_size=10000,
    n_workers=4,
    verbose=False,
):
    """
    Convert CSV to HDF5 (or .npy) using homogeneous 2D arrays.
    Here, we build each dataset as a 2D array (with fixed number of columns)
    so that later you can access columns by index (e.g. jets[:,4] for jet_pt).

    """
    if verbose:
        print(f"Processing {csv_file}...")
        print(f"Processing in parallel using {n_workers} workers...")
    # Schedule processing for each CSV chunk.
    delayed_tasks = []
    start_evt_id = 1
    for chunk in csv_chunk_generator(csv_file, chunk_size):
        delayed_tasks.append(delayed(process_chunk)(chunk, start_evt_id))
        start_evt_id += len(chunk)

    # Compute chunk results in parallel.
    if verbose:
        print(f"Processing {start_evt_id - 1} events in {len(delayed_tasks)} chunks...")
    results = compute(*delayed_tasks, num_workers=n_workers)

    if file_type == "h5":
        h5_filepath = f"{out_path}/{output_prefix}.h5"
        with h5py.File(h5_filepath, "w") as h5file:
            # Create resizable datasets for homogeneous arrays:
            h5file.create_dataset(
                "events",
                shape=(0, 5),
                maxshape=(None, 5),
                dtype=np.float32,
                chunks=True,
                compression="gzip",
            )
            h5file.create_dataset(
                "jets",
                shape=(0, 7),
                maxshape=(None, 7),
                dtype=np.float32,
                chunks=True,
                compression="gzip",
            )
            h5file.create_dataset(
                "constituents",
                shape=(0, 7),
                maxshape=(None, 7),
                dtype=np.float32,
                chunks=True,
                compression="gzip",
            )
            # Append each processed chunk.
            for events_arr, jets_arr, constituents_arr in results:
                if events_arr.size > 0:
                    append_to_hdf5(h5file, "events", events_arr)
                if jets_arr.size > 0:
                    append_to_hdf5(h5file, "jets", jets_arr)
                if constituents_arr.size > 0:
                    append_to_hdf5(h5file, "constituents", constituents_arr)
        if verbose:
            print(f"Data saved to HDF5 file at {h5_filepath}")

    elif file_type == "npy":
        # Accumulate and save as .npy files
        events_list, jets_list, constituents_list = [], [], []
        for events_arr, jets_arr, constituents_arr in results:
            if events_arr.size > 0:
                events_list.append(events_arr)
            if jets_arr.size > 0:
                jets_list.append(jets_arr)
            if constituents_arr.size > 0:
                constituents_list.append(constituents_arr)
        if events_list:
            events_all = np.concatenate(events_list)
            np.save(f"{out_path}/{output_prefix}_events.npy", events_all)
        if jets_list:
            jets_all = np.concatenate(jets_list)
            np.save(f"{out_path}/{output_prefix}_jets.npy", jets_all)
        if constituents_list:
            constituents_all = np.concatenate(constituents_list)
            np.save(f"{out_path}/{output_prefix}_constituents.npy", constituents_all)
        if verbose:
            print(f"Data saved to .npy files with prefix {output_prefix} at {out_path}")
