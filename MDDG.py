#!/usr/bin/env python3
# License: GNU-3.0-or-later
# Copyright (C) 2025
# Original code: Vilmos Neuman, Patryk A. Wesołowski and David J. Wales
# With help from (in alphabetical order): Krzysztof K. Bojarski, Diksha Dewan, Moritz Schäffler and Pamela Smardz

"""
MDDG — Molecular Dynamics to Disconnectivity Graphs
=======================================================

A lightweight Python program to convert molecular dynamics trajectories into disconnectivity graphs.

Authors
-------
Vilmos Neuman, Patryk A. Wesołowski, Krzysztof K. Bojarski, Diksha Dewan
Moritz Schäffler, Pamela Smardz, and David J. Wales

License
-------
This file is part of MDDG and is distributed under the terms of the GNU General Public License v3.0 or later.

Citation
--------
Cite as: V. Neuman, P. A. Wesołowski,K. K. Bojarski, Diksha Dewan, M. Schäffler, P. Smardz, D. J. Wales. " Smoothing Molecular Motion into Energy Landscapes:Disconnectivity Graphs to Visualise Molecular Dynamics " In preparation.

Contributing
------------
Contributions are always welcome!

Usage:
    python MDDG.py --data FILENAME [options]

Required:
    --data FILENAME         Input data file

Options:
    --column COL            Which column to use for energy (default: auto-detect (avoid if possible))
                            Can be: column number (1,2,3...) or name (energy, PotEng, etc.)
    --skip SKIP             Skip first SKIP frames (default: 0)
                            Useful for removing equilibration period
    --step STEP             Sample every STEP frames (default: 1)
    --window WINDOW         Smoothing window size (default: 5, use 0 for no smoothing)
    --polyorder ORDER       Polynomial order for Savitzky-Golay filter (default: 2)
                            Must be less than window size. Higher polyorder --> less smoothing
    --delimiter DELIM       Column delimiter (default: auto-detect (avoid if possible))
    --help                  Show this help message

The script automatically handles:
    - Files with or without headers
    - 2 column format (frame, energy)
    - 3 column format (frame, potential_energy, enthalpy)
    - Various delimiters (spaces, tabs, commas)
    - Different column names

Examples:
    # Auto-detect everything
    python MDDG.py --data trajectory.dat
    
    # Skip first 1000 frames (equilibration) and sample every 10 frames
    python MDDG.py --data trajectory.dat --skip 1000 --step 10
    
    # Specify which column to use by number (1 indexed)
    python MDDG.py --data trajectory.dat --column 3
    
    # Specify column by name
    python MDDG.py --data trajectory.dat --column enthalpy
    
    # Custom polynomial order for smoothing
    python MDDG.py --data trajectory.dat --window 9 --polyorder 3
"""

__version__ = "0.1.1"       
__license__ = "GNU-3.0"


from pathlib import Path
import sys
import numpy as np
from numpy.polynomial import Polynomial
from scipy.signal import savgol_filter
import argparse
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def detect_delimiter(line):
    """Auto detect the delimiter used in the file"""
    # Count occurrences of common delimiters
    delimiters = {
        ',': line.count(','),
        '\t': line.count('\t'),
        ';': line.count(';'),
        '|': line.count('|'),
        ' ': len(line.split()) - 1  
    }
    
    # Return the delimiter with the most occurrences
    if max(delimiters.values()) == 0:
        return None  # whitespace delimited
    return max(delimiters, key=delimiters.get)

def is_header_line(line):
    """Check if a line is likely a header"""
    # Remove common comment characters
    clean_line = line.strip().lstrip('#').lstrip('!')
    
    # Split the line
    parts = clean_line.split()
    
    if not parts:
        return False
    
    # Count how many parts are valid numbers
    num_count = 0
    for part in parts:
        try:
            # Try to parse as float
            val = float(part)
            # Check if it's a valid number 
            if np.isfinite(val):
                num_count += 1
        except (ValueError, OverflowError):
            pass
    
    # If less than half are valid numbers, probably a header
    return num_count < len(parts) / 2

def parse_header(line, delimiter=None):
    """Parse header line to get column names"""
    clean_line = line.strip().lstrip('#').lstrip('!')
    
    if delimiter and delimiter != ' ':
        columns = [col.strip() for col in clean_line.split(delimiter)]
    else:
        columns = clean_line.split()
    
    return columns

def find_energy_column(headers):
    """Auto-detect which column contains energy data"""
    # Common energy column names (case-insensitive)
    energy_patterns = [
        r'eng|energy|pe|pot.*eng|potential',
        r'enthalpy|h\b',  # \b for word boundary
        r'etot|total.*eng',
        
    ]
    
    for i, header in enumerate(headers):
        header_lower = header.lower()
        for pattern in energy_patterns:
            if re.search(pattern, header_lower):
                return i, header
    
    # If no energy column found, assume it's the last numeric column
    return len(headers) - 1, headers[-1] if headers else "Energy"

def check_if_frame_numbers(data_column):
    """Check if a column looks like frame numbers"""
    if len(data_column) < 2:
        return False
    
    # Check if it's sequential or near sequential
    diffs = np.diff(data_column)
    
    # Check for constant spacing (allowing some tolerance for floating point)
    if len(diffs) > 0:
        mean_diff = np.mean(diffs)
        if mean_diff > 0 and np.allclose(diffs, mean_diff, rtol=0.1):
            return True
    
    # Check if it's just 0, 1, 2, 3, ...
    expected = np.arange(len(data_column))
    if np.allclose(data_column, expected):
        return True
    
    return False

def load_data_flexible(filename, column_spec=None, delimiter_spec=None, skip_frames=0):
    """
    Flexibly load data from various file formats
    Returns: frame_numbers, energies, column_name
    """
    print(f"Analyzing file format...")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        raise ValueError("File is empty")
    
    # Find the first non empty, non comment line
    first_data_line_idx = 0
    headers = None
    delimiter = delimiter_spec
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Check if this is a header line
        if is_header_line(line):
            if delimiter is None:
                delimiter = detect_delimiter(line)
            headers = parse_header(line, delimiter)
            print(f"  Detected header: {headers}")
            first_data_line_idx = i + 1
        else:
            # This is data
            if delimiter is None:
                delimiter = detect_delimiter(line)
            first_data_line_idx = i
            break
    
    # Determine delimiter for numpy
    numpy_delimiter = delimiter if delimiter and delimiter != ' ' else None
    
    # Load the numeric data
    try:
        data = np.loadtxt(filename, delimiter=numpy_delimiter, skiprows=first_data_line_idx)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Trying alternative parsing...")
        # Try loading with manual parsing
        data = []
        for line in lines[first_data_line_idx:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if delimiter and delimiter != ' ':
                parts = line.split(delimiter)
            else:
                parts = line.split()
            try:
                data.append([float(p) for p in parts])
            except ValueError:
                continue
        data = np.array(data)
    
    if len(data.shape) == 1:
        # Single column - create frame numbers
        print("  Single column detected - creating frame numbers")
        frame_numbers = np.arange(len(data))
        energies = data
        column_name = "Energy"
    else:
        # Multiple columns
        n_cols = data.shape[1]
        print(f"  Detected {n_cols} columns")
        
        # Check if first column looks like frame numbers
        first_col_is_frames = check_if_frame_numbers(data[:, 0])
        
        if first_col_is_frames:
            print("  First column appears to be frame/step numbers")
            frame_numbers = data[:, 0]  # Keep as float if needed
            data_start_col = 1
        else:
            print("  First column doesn't look like frame numbers, treating it as data")
            frame_numbers = np.arange(len(data))
            data_start_col = 0
        
        # Determine which column to use for energy
        if column_spec is not None:
            # User specified a column
            if isinstance(column_spec, str):
                # Try to parse as number first
                try:
                    col_idx = int(column_spec) - 1  # Convert to 0-indexed
                    # FIX: Add proper bounds checking
                    if col_idx < 0:
                        raise ValueError(f"Column number must be positive (got {column_spec})")
                    if col_idx >= n_cols:
                        raise ValueError(f"Column {column_spec} does not exist (file has {n_cols} columns)")
                    energies = data[:, col_idx]
                    column_name = headers[col_idx] if headers and col_idx < len(headers) else f"Column {col_idx+1}"
                except ValueError as e:
                    # Check if it's because of the bounds check or because it's not a number
                    try:
                        int(column_spec)
                        # It was a number, so re raise the bounds error
                        raise e
                    except ValueError:
                        # It's a column name
                        if headers:
                            # Find column by name
                            found = False
                            for i, h in enumerate(headers):
                                if column_spec.lower() in h.lower():
                                    energies = data[:, i]
                                    column_name = h
                                    found = True
                                    break
                            if not found:
                                raise ValueError(f"Column '{column_spec}' not found in headers: {headers}")
                        else:
                            raise ValueError(f"Column name '{column_spec}' specified but no headers found")
            else:
                col_idx = int(column_spec) - 1
                # FIX: Add proper bounds checking for integer column spec
                if col_idx < 0:
                    raise ValueError(f"Column number must be positive (got {column_spec})")
                if col_idx >= n_cols:
                    raise ValueError(f"Column {column_spec} does not exist (file has {n_cols} columns)")
                energies = data[:, col_idx]
                column_name = headers[col_idx] if headers and col_idx < len(headers) else f"Column {col_idx+1}"
        else:
            # Auto detect energy column
            if n_cols == 2 and first_col_is_frames:
                # Simple 2-column format: frame, energy
                energies = data[:, 1]
                column_name = headers[1] if headers and len(headers) > 1 else "Energy"
            elif n_cols >= 2:
                # Multiple columns - try to find energy column
                if headers:
                    # Search from the appropriate starting column
                    search_headers = headers[data_start_col:]
                    col_idx, column_name = find_energy_column(search_headers)
                    col_idx += data_start_col  # Adjust for starting position
                    energies = data[:, col_idx]
                    print(f"  Auto-selected column '{column_name}' for energy")
                else:
                    # No headers - use first data column after frames (if any)
                    col_idx = data_start_col
                    energies = data[:, col_idx]
                    column_name = f"Column {col_idx + 1}"
                    print(f"  No headers found. Using column {col_idx + 1} as energy.")
                    print(f"  (Use --column to specify a different column)")
            else:
                raise ValueError(f"Unexpected number of columns: {n_cols}")
    
    # Apply skip_frames
    if skip_frames > 0:
        if skip_frames >= len(energies):
            raise ValueError(f"Cannot skip {skip_frames} frames - only {len(energies)} frames available")
        print(f"  Skipping first {skip_frames} frames")
        frame_numbers = frame_numbers[skip_frames:]
        energies = energies[skip_frames:]
    
    print(f"  Using energy data from: {column_name}")
    print(f"  Loaded {len(energies)} data points (after skipping)")
    
    return frame_numbers, energies, column_name

def check_endpoint_via_extrapolation(energies, position='first'):
    """
    Check if endpoint is a minimum by polynomial extrapolation.
    Uses first/last 20 frames (or 1/4 of trajectory if shorter).
    """
    # Need at least 4 points for cubic polynomial
    if len(energies) < 4:
        print(f"    Warning: Too few points ({len(energies)}) for extrapolation check")
        return False
    
    # Use 20 frames or 1/4 of trajectory, whichever is smaller, but at least 4
    n_fit = min(20, max(4, len(energies) // 4))

    if position == 'first':
        # Fit first n_fit points
        x = np.arange(n_fit)
        y = energies[:n_fit]

        # Extrapolate 3 points backwards and check
        x_test = np.array([-3, -2, -1, 0, 1])  # Include frame 0 and 1 for comparison

    else:  # 'last'
        # Fit last n_fit points
        x = np.arange(n_fit)
        y = energies[-n_fit:]

        # Extrapolate 3 points forward and check
        x_test = np.array([n_fit-2, n_fit-1, n_fit, n_fit+1, n_fit+2])

    try:
        # Fit cubic polynomial
        poly = Polynomial.fit(x, y, deg=3)
        y_test = poly(x_test)

        if position == 'first':
            # Is frame 0 (index 3 in x_test) a local min?
            return y_test[3] < y_test[2] and y_test[3] < y_test[4]
        else:
            # Is last frame (index 1 in x_test) a local min?
            return y_test[1] < y_test[0] and y_test[1] < y_test[2]
    except (np.linalg.LinAlgError, ValueError) as e:
        # Polynomial fitting can fail
        print(f"    Warning: Polynomial fitting failed for {position} endpoint: {e}")
        return False

# -------------------------------------------------------------------- PARSE ARGS
parser = argparse.ArgumentParser(
    description='Generate min.data and ts.data from energy data',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__
)

parser.add_argument('--data', type=str, required=True,
                    help='Input data file (REQUIRED)')
parser.add_argument('--column', type=str, default=None,
                    help='Which column to use (number or name)')
parser.add_argument('--skip', type=int, default=0,
                    help='Skip first SKIP frames (default: 0)')
parser.add_argument('--step', type=int, default=1,
                    help='Sample every STEP frames (default: 1)')
parser.add_argument('--window', type=int, default=5,
                    help='Smoothing window size (default: 5, use 0 for no smoothing)')
parser.add_argument('--polyorder', type=int, default=2,
                    help='Polynomial order for Savitzky-Golay filter (default: 2)')
parser.add_argument('--delimiter', type=str, default=None,
                    help='Column delimiter (default: auto-detect)')

args = parser.parse_args()

# Set parameters from arguments
SKIP_FRAMES = args.skip
STEP_SIZE = args.step
WINDOW_SIZE = args.window
POLYORDER = args.polyorder
DATA_FILE = Path(args.data)
COLUMN_SPEC = args.column
DELIMITER = args.delimiter

# Validate arguments
if SKIP_FRAMES < 0:
    sys.exit("Error: Skip value cannot be negative")

if STEP_SIZE < 1:
    sys.exit("Error: Step size must be at least 1")

if WINDOW_SIZE < 0:
    sys.exit("Error: Window size cannot be negative")

if WINDOW_SIZE > 0 and WINDOW_SIZE < 3:
    print(f"Warning: Window size {WINDOW_SIZE} is too small for effective smoothing. Using 3.")
    WINDOW_SIZE = 3

if POLYORDER < 0:
    sys.exit("Error: Polynomial order cannot be negative")

# Check if data file exists
if not DATA_FILE.exists():
    sys.exit(f"Error: Data file '{DATA_FILE}' not found")

# -------------------------------------------------------------------- PARSE DATA
print(f"Reading data from: {DATA_FILE}")
print(f"Parameters:")
if SKIP_FRAMES > 0:
    print(f"  - Skipping first {SKIP_FRAMES} frames")
print(f"  - Sampling every {STEP_SIZE} frame(s)")
print(f"  - Smoothing window: {WINDOW_SIZE if WINDOW_SIZE > 0 else 'disabled'}")
if WINDOW_SIZE > 0:
    print(f"  - Polynomial order: {POLYORDER}")
if COLUMN_SPEC:
    print(f"  - Using column: {COLUMN_SPEC}")
print()

try:
    frame_numbers, all_energies, energy_label = load_data_flexible(
        DATA_FILE, COLUMN_SPEC, DELIMITER, SKIP_FRAMES
    )
except Exception as e:
    sys.exit(f"Error loading data: {e}")

total_frames = len(all_energies)

if total_frames == 0:
    sys.exit(f"No energy data found in the file")

# Quick statistics
print(f"\nData statistics (after skipping):")
print(f"  - Energy range: [{all_energies.min():.4f}, {all_energies.max():.4f}]")
print(f"  - Mean energy: {all_energies.mean():.4f}")
# ----------------------------------------------------------------- SMOOTHING
# FIX: Use consistent variable for polyorder throughout
actual_polyorder = POLYORDER  # Initialise with the original value

if WINDOW_SIZE > 0:
    print(f"\nSmoothing trajectory ({total_frames} frames) with window size {WINDOW_SIZE}...")
    
    # Ensure window size is odd for Savitzky-Golay FIRST
    window = WINDOW_SIZE if WINDOW_SIZE % 2 == 1 else WINDOW_SIZE + 1
    if window != WINDOW_SIZE:
        print(f"  Note: Adjusted window size to {window} (must be odd for SG filter)")
    
    # Now check if we have enough frames for the ADJUSTED window
    if total_frames > window:  # Check against adjusted window
        # Validate and adjust polynomial order if necessary
        if window <= POLYORDER:
            actual_polyorder = window - 1
            print(f"  Warning: Polynomial order ({POLYORDER}) must be < window size ({window}).")
            print(f"           Adjusting to {actual_polyorder}")
        else:
            actual_polyorder = POLYORDER
        
        # Apply Savitzky-Golay filter
        try:
            all_smoothed_energies = savgol_filter(all_energies, window, actual_polyorder)
            print(f"  Applied Savitzky-Golay filter (window={window}, polyorder={actual_polyorder})")
        except Exception as e:
            print(f"  Error in smoothing: {e}")
            print("  Using original data without smoothing")
            all_smoothed_energies = all_energies
    else:
        print(f"  Warning: Trajectory ({total_frames} frames) not long enough for window size ({window})")
        print("  Using original data without smoothing")
        all_smoothed_energies = all_energies

# ----------------------------------------------------------------- SAMPLING
print(f"\nSampling every {STEP_SIZE} frame(s)...")

# Apply sampling to all data
sampled_indices = list(range(0, total_frames, STEP_SIZE))
energies = all_energies[sampled_indices]  # Keep original for reference
smoothed_energies = all_smoothed_energies[sampled_indices]  # Use smoothed for analysis
sampled_frame_numbers = frame_numbers[sampled_indices]  # Keep track of actual frame numbers
n_frames = len(sampled_indices)

print(f"  Using {n_frames} frames after sampling")

# Ensure we have enough frames to proceed
if n_frames < 3:
    sys.exit(f"Error: Too few frames ({n_frames}) after sampling. Need at least 3 frames.")

# ----------------------------------------------------------- IDENTIFY MINIMA
print("\nIdentifying minima...")

# Find interior minima first
minima_frames = []
for i in range(1, n_frames - 1):
    # Local minimum: lower than both neighbors
    if smoothed_energies[i] < smoothed_energies[i-1] and smoothed_energies[i] < smoothed_energies[i+1]:
        minima_frames.append(i)

# Check endpoints via extrapolation
if 0 not in minima_frames:
    if check_endpoint_via_extrapolation(smoothed_energies, position='first'):
        minima_frames.insert(0, 0)
        print(f"  First frame included as minimum (extrapolation check)")

if n_frames - 1 not in minima_frames:
    if check_endpoint_via_extrapolation(smoothed_energies, position='last'):
        minima_frames.append(n_frames - 1)
        print(f"  Last frame included as minimum (extrapolation check)")

print(f"  Found {len(minima_frames)} minima")

if len(minima_frames) == 0:
    print("\nWarning: No minima found! This might indicate:")
    print("  - The trajectory is monotonic (always increasing/decreasing)")
    print("  - The smoothing window is too large")
    print("  - The sampling step is too large")
    print("\nCreating minimal output with endpoints only...")
    minima_frames = [0, n_frames - 1]

# ---------------------------------------------------- PROCESS MINIMA
minima_list = [(frame, smoothed_energies[frame], sampled_frame_numbers[frame]) for frame in minima_frames]
minimum_mapping = {frame: idx + 1 for idx, frame in enumerate(minima_frames)}

# --------------------------------------------------------- FIND TRANSITIONS
print("Finding transition states and connectivity...")

transitions = []
current_minimum = None
last_minimum_frame = None

for i in range(n_frames):
    if i in minima_frames:
        new_minimum = minimum_mapping[i]
        
        if current_minimum is not None and new_minimum != current_minimum:
            # Found a transition
            start = last_minimum_frame
            end = i
            
            if start < end:
                segment = smoothed_energies[start:end+1]
                ts_relative_idx = np.argmax(segment)
                ts_frame = start + ts_relative_idx
                ts_energy = smoothed_energies[ts_frame]
                
                transitions.append((current_minimum, new_minimum, ts_frame, ts_energy))
        
        current_minimum = new_minimum
        last_minimum_frame = i

print(f"  Found {len(transitions)} transitions")

# ----------------------------------------------------------------- MIN.DATA
print("\nWriting min.data...")
with open("min.data", "w") as fmin:
    for idx, (frame, energy, actual_frame_num) in enumerate(minima_list, start=1):
        fmin.write(f"{energy:20.10f}   0.0   1   0.0   0.0   0.0\n")

# -------------------------------------------------------- COORDINATE MAPPING
print("Writing minimum_frames.txt...")
with open("minimum_frames.txt", "w") as fmap:
    fmap.write("# Mapping of minima to frame numbers for coordinate extraction\n")
    fmap.write(f"# Data column: {energy_label}\n")
    if SKIP_FRAMES > 0:
        fmap.write(f"# Skipped first {SKIP_FRAMES} frames\n")
    fmap.write(f"# Sampling: every {STEP_SIZE} frames\n")
    fmap.write(f"# Smoothing window: {WINDOW_SIZE if WINDOW_SIZE > 0 else 'none'}\n")
    if WINDOW_SIZE > 0:
        fmap.write(f"# Polynomial order: {actual_polyorder}\n")  # FIX: Use consistent variable
    fmap.write("# Format: Minimum_ID | Sampled_Index | Actual_Frame_Number | Energy_Value\n")
    fmap.write("#" + "-"*70 + "\n\n")
    
    for idx, (frame, energy, actual_frame_num) in enumerate(minima_list, start=1):
        # Handle both int and float frame numbers gracefully
        if isinstance(actual_frame_num, (int, np.integer)):
            fmap.write(f"Minimum {idx:4d} | Index {frame:6d} | Frame {actual_frame_num:10d} | E = {energy:10.4f}\n")
        else:
            fmap.write(f"Minimum {idx:4d} | Index {frame:6d} | Frame {actual_frame_num:10.4f} | E = {energy:10.4f}\n")

# ------------------------------------------------------------------ TS.DATA
print("Writing ts.data...")
with open("ts.data", "w") as fts:
    for from_idx, to_idx, ts_frame, ts_energy in transitions:
        # Use the actual TS energy without artificial adjustment
        # The TS is already the maximum between the two minima by construction
        fts.write(f"{ts_energy:20.10f}   0.0   1   {from_idx:5d}   {to_idx:5d}   0.0   0.0   0.0\n")

# ------------------------------------------------------------------ SUMMARY
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Input file: {DATA_FILE}")
print(f"Data column: {energy_label}")
print(f"Total frames in file: {total_frames + SKIP_FRAMES:,}")
if SKIP_FRAMES > 0:
    print(f"Frames skipped: {SKIP_FRAMES:,}")
print(f"Frames used: {total_frames:,}")
print(f"Sampled frames: {n_frames:,} (every {STEP_SIZE})")
if WINDOW_SIZE > 0:
    print(f"Smoothing: window={WINDOW_SIZE}, polyorder={actual_polyorder}")  # FIX: Use consistent variable
else:
    print(f"Smoothing: disabled")
print(f"Minima found: {len(minima_list)}")
print(f"Transitions found: {len(transitions)}")
print(f"\nOutput files:")
print(f"  - min.data (minima energies)")
print(f"  - ts.data (transition states)")
print(f"  - minimum_frames.txt (frame mapping)")

# -------------------------------------------------------------------- PLOT
print("\nCreating visualization...")

fig, ax = plt.subplots(figsize=(12, 6))

# Limit points for plotting if trajectory is very long
max_plot_points = 5000
if n_frames > max_plot_points:
    plot_stride = n_frames // max_plot_points
else:
    plot_stride = 1

plot_indices = np.arange(0, n_frames, plot_stride)
plot_frame_numbers = sampled_frame_numbers[plot_indices]
plot_energies = energies[plot_indices]

# Plot raw and smoothed energies
if WINDOW_SIZE > 0:
    plot_smoothed = smoothed_energies[plot_indices]
    ax.plot(plot_frame_numbers, plot_energies, 'lightgray', alpha=0.6, linewidth=0.8, label='Raw')
    ax.plot(plot_frame_numbers, plot_smoothed, 'darkblue', linewidth=1.2, label='Smoothed')
else:
    ax.plot(plot_frame_numbers, plot_energies, 'darkblue', linewidth=1.0)

# Simple formatting
ax.set_xlabel('Frame Number', fontsize=12)
ax.set_ylabel(energy_label, fontsize=12)
title_text = f'Energy Trajectory ({len(minima_list)} minima, {len(transitions)} transitions)'
if SKIP_FRAMES > 0:
    title_text += f' - Skipped first {SKIP_FRAMES} frames'
ax.set_title(title_text, fontsize=14)
ax.grid(True, alpha=0.3)
if WINDOW_SIZE > 0:
    ax.legend()

plt.tight_layout()
plt.savefig('energy_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('energy_analysis.pdf', bbox_inches='tight')
plt.close()

print("  Created energy_analysis.png and energy_analysis.pdf")

