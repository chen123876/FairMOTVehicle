#!/usr/bin/env python3

import os
import os.path as osp
import numpy as np


def convert_subwaycar_detections():
    """Convert subwaycar_c.txt MOT format to FairMOT det.txt format"""
    
    # Paths
    input_file = "/2411041025/2513041010/FairMOTVehicle/dataset/SubwayCar/test/subwaycar_c/det/subwaycar_c.txt"
    output_file = "/2411041025/2513041010/FairMOTVehicle/dataset/SubwayCar/test/subwaycar_c/det/det.txt"
    
    # Image dimensions
    seq_width = 1920
    seq_height = 1080
    
    print(f"Converting: {input_file}")
    print(f"To: {output_file}")
    print(f"Image size: {seq_width}x{seq_height}")
    
    # Check input
    if not osp.exists(input_file):
        print(f"Error: Input file not found")
        return False
    
    # Read data
    try:
        data = np.loadtxt(input_file, delimiter=',')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        print(f"Read {data.shape[0]} detections")
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    # Process data
    output_lines = []
    tid_counter = 1  # Start from 1 (similar to reference code)
    
    for row in data:
        if len(row) < 7:
            continue
        
        # Parse fields
        frame = int(row[0])
        orig_tid = int(row[1])
        x = float(row[2])
        y = float(row[3])
        w = float(row[4])
        h = float(row[5])
        conf = float(row[6])
        
        # Skip if confidence is too low (optional)
        # if conf < 0.5: continue
        
        # Calculate center coordinates
        cx = x + w / 2
        cy = y + h / 2
        
        # Normalize
        cx_norm = cx / seq_width
        cy_norm = cy / seq_height
        w_norm = w / seq_width
        h_norm = h / seq_height
        
        # FairMOT format: class_id target_id cx_norm cy_norm w_norm h_norm
        # Use sequential IDs for detections
        line = f"0 {tid_counter} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
        output_lines.append(line)
        
        tid_counter += 1
    
    # Write output
    output_dir = osp.dirname(output_file)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
    
    print(f"\nSuccessfully converted {len(output_lines)} detections")
    print(f"Output written to: {output_file}")
    
    # Show sample
    if output_lines:
        print("\nSample output (first 3 lines):")
        for i in range(min(3, len(output_lines))):
            print(f"  {output_lines[i]}")
    
    return True


if __name__ == "__main__":
    convert_subwaycar_detections()