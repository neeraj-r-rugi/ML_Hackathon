#!/usr/bin/env python3
"""
PathSnap - Command-Line Terminal EDA & Graph Visualizer
=====================================================
Renders high-resolution ASCII & Unicode graphs, histograms, and 2D spatial maps
directly in the terminal using ANSI colors and Unicode block elements.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# ANSI Color Codes
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
RED = "\033[31m"
GRAY = "\033[90m"


def render_ascii_histogram(
    data: np.ndarray, 
    title: str, 
    unit: str = "", 
    bins: int = 12, 
    max_bar_width: int = 36,
    color: str = GREEN
) -> None:
    """Render a clean horizontal bar histogram in the terminal."""
    clean_data = data[~np.isnan(data)]
    if len(clean_data) == 0:
        print(f"{YELLOW}[!] No valid data for {title}{RESET}")
        return
        
    counts, edges = np.histogram(clean_data, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1
    total = len(clean_data)
    
    mean_val = float(np.mean(clean_data))
    median_val = float(np.median(clean_data))
    std_val = float(np.std(clean_data))
    min_val = float(np.min(clean_data))
    max_val = float(np.max(clean_data))

    print(f"\n{BOLD}{CYAN}┌{'─'*66}┐{RESET}")
    print(f"{BOLD}{CYAN}│ {title.center(64)} │{RESET}")
    print(f"{BOLD}{CYAN}├{'─'*66}┤{RESET}")
    print(f"│ {GRAY}Min: {min_val:.1f}{unit} | Mean: {mean_val:.1f}{unit} | Median: {median_val:.1f}{unit} | Max: {max_val:.1f}{unit} | N={total:,}{RESET}".ljust(75) + f"{CYAN}│{RESET}")
    print(f"{BOLD}{CYAN}├{'─'*22}┬{'─'*(max_bar_width+2)}┬{'─'*17}┤{RESET}")
    
    for i in range(len(counts)):
        low = edges[i]
        high = edges[i+1]
        c = counts[i]
        pct = (c / total) * 100
        
        # Unicode block bar rendering
        bar_len = int((c / max_count) * max_bar_width)
        bar_str = "█" * bar_len
        
        range_str = f"[{low:6.1f} - {high:6.1f}{unit}]"
        stat_str = f"{c:6,d} ({pct:4.1f}%)"
        print(f"│ {YELLOW}{range_str:<20}{RESET} │ {color}{bar_str:<{max_bar_width}}{RESET} │ {stat_str:>15} │")
        
    print(f"{BOLD}{CYAN}└{'─'*22}┴{'─'*(max_bar_width+2)}┴{'─'*17}┘{RESET}")


def render_ascii_2d_map(
    track_df: pd.DataFrame, 
    route_arcs_df: pd.DataFrame = None, 
    width: int = 65, 
    height: int = 22
) -> None:
    """
    Renders an ASCII 2D spatial canvas projecting GPS points and ground-truth route
    directly in the command line terminal.
    """
    print(f"\n{BOLD}{MAGENTA}┌{'─'*width}┐{RESET}")
    print(f"{BOLD}{MAGENTA}│ {'2D SPATIAL MAP: GPS OBSERVED PATH (·) vs ROUTE (═)'.center(width-2)} │{RESET}")
    print(f"{BOLD}{MAGENTA}└{'─'*width}┘{RESET}")
    
    # Coordinates in projected meters or lon/lat
    x_col = "x_m" if "x_m" in track_df.columns else "longitude"
    y_col = "y_m" if "y_m" in track_df.columns else "latitude"
    
    xs = track_df[x_col].values
    ys = track_df[y_col].values
    
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    
    # Initialize character grid
    grid = [[" " for _ in range(width)] for _ in range(height)]
    
    # Helper to map coordinates to grid
    def to_grid(x, y):
        col = int(((x - min_x) / span_x) * (width - 1))
        row = int(((y - min_y) / span_y) * (height - 1))
        # Invert row for standard Cartesian (0 at bottom)
        row = (height - 1) - max(0, min(row, height - 1))
        col = max(0, min(col, width - 1))
        return row, col

    # Draw GPS points
    for x, y in zip(xs, ys):
        r, c = to_grid(x, y)
        grid[r][c] = f"{BLUE}·{RESET}"
        
    # Draw Ground-Truth Route Arcs if provided
    if route_arcs_df is not None and not route_arcs_df.empty:
        fx_col = "from_x" if "from_x" in route_arcs_df.columns else "longitude"
        fy_col = "from_y" if "from_y" in route_arcs_df.columns else "latitude"
        tx_col = "to_x" if "to_x" in route_arcs_df.columns else "longitude"
        ty_col = "to_y" if "to_y" in route_arcs_df.columns else "latitude"
        
        for _, arc in route_arcs_df.iterrows():
            r1, c1 = to_grid(arc[fx_col], arc[fy_col])
            r2, c2 = to_grid(arc[tx_col], arc[ty_col])
            # Simple Bresenham interpolation on grid
            n_steps = max(abs(r2 - r1), abs(c2 - c1), 1)
            for step in range(n_steps + 1):
                interp_r = int(r1 + (r2 - r1) * (step / n_steps))
                interp_c = int(c1 + (c2 - c1) * (step / n_steps))
                interp_r = max(0, min(interp_r, height - 1))
                interp_c = max(0, min(interp_c, width - 1))
                grid[interp_r][interp_c] = f"{GREEN}═{RESET}"
                
    # Mark start & end
    start_r, start_c = to_grid(xs[0], ys[0])
    end_r, end_c = to_grid(xs[-1], ys[-1])
    grid[start_r][start_c] = f"{BOLD}{GREEN}S{RESET}"
    grid[end_r][end_c] = f"{BOLD}{RED}E{RESET}"
    
    # Border printing
    print(f"{GRAY}┌" + "─" * width + "┐" + RESET)
    for row in grid:
        print(f"{GRAY}│{RESET}" + "".join(row) + f"{GRAY}│{RESET}")
    print(f"{GRAY}└" + "─" * width + "┘" + RESET)
    print(f"Legend: {BOLD}{GREEN}S{RESET}=Start | {BOLD}{RED}E{RESET}=End | {BLUE}·{RESET}=GPS Ping | {GREEN}═{RESET}=Ground-Truth Route Arc\n")


def run_cli_eda(trip_id: str = "00000001", split: str = "train"):
    """Run full command-line EDA for a selected trip."""
    manifest_path = Path("processed/trip_manifest.csv")
    if not manifest_path.exists():
        print(f"{RED}[!] processed/trip_manifest.csv not found!{RESET}")
        return
        
    manifest = pd.read_csv(manifest_path)
    print(f"\n{BOLD}{GREEN}{'='*68}{RESET}")
    print(f"{BOLD}{GREEN}   PATHSNAP COMMAND-LINE EXPLORATORY DATA ANALYSIS (EDA){RESET}")
    print(f"{BOLD}{GREEN}{'='*68}{RESET}")
    
    # 1. Dataset Partition Summary Table
    print(f"\n{BOLD}{CYAN}1. PARTITION SUMMARY (from processed/trip_manifest.csv):{RESET}")
    p_summary = manifest.groupby("split").agg({
        "trip_id": "count",
        "n_track": ["sum", "mean", "min", "max"],
        "n_arcs": ["sum", "mean", "min", "max"],
        "n_route": ["sum", "mean", "min", "max"]
    }).round(1)
    print(p_summary.to_string())
    
    # 2. Inspect Target Trip
    trip_dir = Path("processed") / split / trip_id
    if not trip_dir.exists():
        # Fallback to first available trip
        available = list((Path("processed") / split).glob("*"))
        if available:
            trip_dir = available[0]
            trip_id = trip_dir.name
            
    print(f"\n{BOLD}{CYAN}2. SELECTED TRIP DETAILS: {trip_id} (Partition: {split}){RESET}")
    track_file = trip_dir / "track_features.csv"
    arcs_file = trip_dir / "arcs_features.csv"
    route_file = trip_dir / "route.csv"
    cand_file = trip_dir / "candidate_features.csv"
    
    track_df = pd.read_csv(track_file)
    arcs_df = pd.read_csv(arcs_file)
    route_df = pd.read_csv(route_file) if route_file.exists() else pd.DataFrame()
    cand_df = pd.read_csv(cand_file) if cand_file.exists() else pd.DataFrame()
    
    # Filter route arcs
    route_arcs_df = arcs_df[arcs_df["arc_id"].isin(route_df["arc_id"].values)] if not route_df.empty else None

    # Print Key Metrics Card
    dt_vals = track_df["dt"].values[1:]
    is_1s = np.mean(np.isclose(dt_vals, 1.0, atol=0.05)) * 100 if len(dt_vals) > 0 else 100.0
    
    print(f"{GRAY}┌{'─'*66}┐{RESET}")
    print(f"│ • Total GPS Track Pings    : {len(track_df):<6,d} | Duration: {track_df['timestamp'].iloc[-1]-track_df['timestamp'].iloc[0]:.0f} s ({((track_df['timestamp'].iloc[-1]-track_df['timestamp'].iloc[0])/60):.1f} min)")
    print(f"│ • Total Road Network Arcs  : {len(arcs_df):<6,d} | Route Arcs: {len(route_df):<6,d}")
    print(f"│ • Route Compression Ratio  : {len(route_df)/max(1, len(track_df)):.3f} (Traversed arcs per GPS ping)")
    print(f"│ • 1-Second Sampling Rate   : {is_1s:.1f}% | Time Gaps (>2s): {int(track_df['is_time_gap'].sum())}")
    print(f"│ • Speed Outliers (>70m/s)  : {int(track_df['is_speed_outlier'].sum())} | Max Speed: {track_df['speed_kmh'].max():.1f} km/h")
    print(f"{GRAY}└{'─'*66}┘{RESET}")
    
    # 3. Terminal Graphs
    # Graph 1: 2D Spatial Map
    render_ascii_2d_map(track_df, route_arcs_df, width=66, height=20)
    
    # Graph 2: Vehicle Speed Distribution (km/h)
    render_ascii_histogram(
        track_df["speed_kmh"].values, 
        title="Vehicle Speed Distribution", 
        unit=" km/h", 
        bins=10, 
        color=GREEN
    )
    
    # Graph 3: Sampling Interval Distribution (dt)
    render_ascii_histogram(
        track_df["dt"].values[1:], 
        title="GPS Sampling Interval (dt)", 
        unit=" s", 
        bins=8, 
        color=BLUE
    )
    
    # Graph 4: Road Arc Length Distribution (m)
    render_ascii_histogram(
        arcs_df["length_m"].values, 
        title="Road Arc Length Distribution", 
        unit=" m", 
        bins=10, 
        color=YELLOW
    )
    
    # Graph 5: Candidate Arcs per GPS Observation
    if not cand_df.empty:
        c_counts = cand_df.groupby("gps_index")["arc_id"].count().values
        render_ascii_histogram(
            c_counts, 
            title="Candidate Arcs per GPS Point (75m Search Radius)", 
            unit=" arcs", 
            bins=8, 
            color=MAGENTA
        )
        
        # Graph 6: Perpendicular Distance to Candidate Road
        render_ascii_histogram(
            cand_df["distance_to_arc_m"].values, 
            title="Point-to-Road Perpendicular Distance", 
            unit=" m", 
            bins=10, 
            color=CYAN
        )

    print(f"\n{BOLD}{GREEN}[✓] Command-line EDA complete for Trip {trip_id}!{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="PathSnap Command-Line EDA Visualizer")
    parser.add_argument("--trip-id", type=str, default="00000001", help="Trip ID to visualize")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Partition (train or test)")
    args = parser.parse_args()
    
    run_cli_eda(trip_id=args.trip_id, split=args.split)


if __name__ == "__main__":
    main()
