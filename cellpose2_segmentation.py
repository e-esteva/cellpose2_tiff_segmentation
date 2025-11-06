#!/usr/bin/env python3
"""
Cellpose segmentation script for single TIFF files with intelligent tiling.
Automatically splits large images, segments tiles with overlap, and stitches results.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tifffile import imread, imwrite
from cellpose import models
from scipy import ndimage
import json
from datetime import datetime
import xml.etree.ElementTree as ET


def get_available_gpu_memory():
    """Estimate available GPU memory in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory*torch.cuda.device_count() / 1e9
            return gpu_mem * 0.8  # Use 80% of available memory
    except:
        pass
    return 4.0  # Conservative default: 4GB


def get_ome_channel_names(tiff_path):
    """Extract channel names from OME-TIFF metadata."""
    from tifffile import TiffFile
    
    with TiffFile(tiff_path) as tif:
        if tif.ome_metadata:
            root = ET.fromstring(tif.ome_metadata)
            
            namespaces = [
                {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'},
                {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2015-01'},
                {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2013-06'},
            ]
            
            for ns in namespaces:
                channels = root.findall('.//ome:Channel', ns)
                if channels:
                    channel_names = []
                    for ch in channels:
                        name = ch.get('Name', ch.get('ID', f'Channel_{len(channel_names)}'))
                        channel_names.append(name)
                    return channel_names
    
    return None


def get_ome_resolution(tiff_path):
    """Extract physical resolution (microns/pixel) from OME-TIFF metadata."""
    from tifffile import TiffFile
    
    with TiffFile(tiff_path) as tif:
        if tif.ome_metadata:
            root = ET.fromstring(tif.ome_metadata)
            
            for ns_url in ['http://www.openmicroscopy.org/Schemas/OME/2016-06',
                          'http://www.openmicroscopy.org/Schemas/OME/2015-01']:
                ns = {'ome': ns_url}
                pixels = root.find('.//ome:Pixels', ns)
                
                if pixels is not None:
                    x_res = pixels.get('PhysicalSizeX')
                    y_res = pixels.get('PhysicalSizeY')
                    
                    if x_res and y_res:
                        return {
                            'x_resolution': float(x_res),
                            'y_resolution': float(y_res),
                            'unit': pixels.get('PhysicalSizeXUnit', 'µm')
                        }
    
    return None


def calculate_tile_size(img_shape, diameter, available_memory_gb=None):
    """
    Calculate optimal tile size based on image size and available memory.
    
    Returns:
    --------
    tile_size : int or None
        Optimal tile size (square), or None if no tiling needed
    overlap : int or None
        Overlap between tiles in pixels, or None if no tiling needed
    """
    if available_memory_gb is None:
        available_memory_gb = get_available_gpu_memory()
        print(f"Auto-detected GPU memory: {available_memory_gb:.1f} GB")
    else:
        print(f"Using specified memory: {available_memory_gb:.1f} GB")
    
    # Determine image dimensions
    if len(img_shape) == 3:
        channels, height, width = img_shape
    else:
        height, width = img_shape
        channels = 1
    
    # Calculate image memory requirements
    image_memory_gb = (height * width * channels * 4) / 1e9  # float32
    
    # Cellpose memory overhead: ~3-5x during processing (conservative estimate)
    processing_overhead = 4.0
    estimated_peak_memory_gb = image_memory_gb * processing_overhead
    
    print(f"\nMemory analysis:")
    print(f"  Image dimensions: {height} × {width} × {channels} channels")
    print(f"  Image memory: {image_memory_gb:.2f} GB")
    print(f"  Estimated peak memory (with {processing_overhead}x overhead): {estimated_peak_memory_gb:.2f} GB")
    print(f"  Available memory: {available_memory_gb:.2f} GB")
    
    # Check if tiling is needed
    if estimated_peak_memory_gb <= available_memory_gb * 0.8:  # Use 80% as safety margin
        print(f"  → Image fits in memory - NO TILING REQUIRED")
        return None, None
    
    print(f"  → Image too large - TILING REQUIRED")
    
    # Calculate tile size that fits in memory
    max_tile_area = (available_memory_gb * 0.8 * 1e9) / (channels * 4 * processing_overhead)
    max_tile_size = int(np.sqrt(max_tile_area))
    
    # Set practical limits and round to nice numbers
    max_tile_size = min(16384, max_tile_size)
    min_tile_size = 2048
    
    if max_tile_size < min_tile_size:
        raise ValueError(
            f"Insufficient memory: need at least {(min_tile_size**2 * channels * 4 * processing_overhead) / 1e9:.1f} GB "
            f"but only {available_memory_gb:.1f} GB available"
        )
    
    # Choose tile size (prefer powers of 2)
    for size in [16384, 8192, 4096, 2048]:
        if size <= max_tile_size:
            tile_size = size
            break
    else:
        tile_size = min_tile_size
    
    # Calculate overlap: max(10% of tile, 2*diameter)
    overlap = max(int(tile_size * 0.1), 2 * diameter, 100)
    
    print(f"\nTile configuration:")
    print(f"  Tile size: {tile_size} × {tile_size}")
    print(f"  Overlap: {overlap} pixels")
    
    # Estimate number of tiles
    stride = tile_size - overlap
    num_tiles_y = int(np.ceil((height - overlap) / stride))
    num_tiles_x = int(np.ceil((width - overlap) / stride))
    total_tiles = num_tiles_y * num_tiles_x
    print(f"  Estimated tiles: {num_tiles_y} × {num_tiles_x} = {total_tiles} tiles")
    
    return tile_size, overlap


def generate_tiles(height, width, tile_size, overlap):
    """Generate tile coordinates with overlap."""
    tiles = []
    stride = tile_size - overlap
    tile_idx = 0
    
    y_positions = list(range(0, height - overlap, stride))
    if y_positions[-1] + tile_size < height:
        y_positions.append(height - tile_size)
    
    x_positions = list(range(0, width - overlap, stride))
    if x_positions[-1] + tile_size < width:
        x_positions.append(width - tile_size)
    
    for y_start in y_positions:
        for x_start in x_positions:
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)
            
            # Adjust start if we hit the edge
            if y_end == height:
                y_start = max(0, height - tile_size)
            if x_end == width:
                x_start = max(0, width - tile_size)
            
            tiles.append((y_start, y_end, x_start, x_end, tile_idx))
            tile_idx += 1
    
    return tiles


def calculate_iou(mask1, mask2):
    """Calculate Intersection over Union between two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def merge_tiles(tile_results, tiles_coords, img_shape, overlap, iou_threshold=0.5):
    """
    Merge segmentation results from overlapping tiles using IoU-based duplicate detection.
    """
    height, width = img_shape
    merged_masks = np.zeros((height, width), dtype=np.uint32)
    
    next_cell_id = 1
    
    print("\nMerging tiles with duplicate detection...")
    
    for tile_idx, result in enumerate(tile_results):
        y_start, y_end, x_start, x_end = result['coords']
        tile_masks = result['masks']
        
        tile_cell_ids = np.unique(tile_masks)
        tile_cell_ids = tile_cell_ids[tile_cell_ids != 0]
        
        for local_id in tile_cell_ids:
            cell_mask = tile_masks == local_id
            
            # Check overlap with existing cells
            y_slice = slice(y_start, y_end)
            x_slice = slice(x_start, x_end)
            
            existing_region = merged_masks[y_slice, x_slice]
            overlapping_ids = np.unique(existing_region[cell_mask])
            overlapping_ids = overlapping_ids[overlapping_ids != 0]
            
            # Check for duplicates based on IoU
            is_duplicate = False
            for existing_id in overlapping_ids:
                existing_mask = existing_region == existing_id
                iou = calculate_iou(cell_mask, existing_mask)
                
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                global_id = next_cell_id
                merged_masks[y_slice, x_slice][cell_mask] = global_id
                next_cell_id += 1
        
        if (tile_idx + 1) % 10 == 0:
            print(f"  Merged {tile_idx + 1}/{len(tile_results)} tiles")
    
    print(f"  Total cells after merging: {next_cell_id - 1}")
    
    return merged_masks


def extract_cell_features(img, masks):
    """Extract bounding boxes and mean channel intensities for each cell."""
    if masks.ndim == 3:
        masks = masks.squeeze()
    
    # Handle channel dimension
    if img.ndim == 3 and img.shape[0] <= 10:  # (C, H, W) format
        num_channels = img.shape[0]
        img_chw = img
    elif img.ndim == 3:  # (H, W, C) format
        num_channels = img.shape[2]
        img_chw = np.moveaxis(img, -1, 0)
    elif img.ndim == 2:
        num_channels = 1
        img_chw = img[np.newaxis, :, :]
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")
    
    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids != 0]
    num_cells = len(cell_ids)
    
    print(f"Extracting features for {num_cells} cells with {num_channels} channels...")
    
    features = np.zeros((num_cells, 4 + num_channels), dtype=np.float32)
    
    # Compute mean intensities
    for ch in range(num_channels):
        channel_means = ndimage.mean(img_chw[ch], labels=masks, index=cell_ids)
        features[:, 4 + ch] = channel_means
        if (ch + 1) % 2 == 0 or ch == num_channels - 1:
            print(f"  Processed {ch + 1}/{num_channels} channels")
    
    # Compute bounding boxes
    print("  Computing bounding boxes...")
    slice_objects = ndimage.find_objects(masks)
    
    for idx, cell_id in enumerate(cell_ids):
        if slice_objects[cell_id - 1] is not None:
            slice_y, slice_x = slice_objects[cell_id - 1]
            features[idx, 0] = slice_x.start  # min_x
            features[idx, 1] = slice_x.stop - 1  # max_x
            features[idx, 2] = slice_y.start  # min_y
            features[idx, 3] = slice_y.stop - 1  # max_y
        
        if (idx + 1) % 100000 == 0:
            print(f"    Processed {idx + 1}/{num_cells} cells")
    
    return features


def save_outputs(masks, flows, styles, diams, features, output_dir, filename_base, 
                 channel_names, metadata):
    """Save all segmentation outputs."""
    output_dir = Path(output_dir)
    
    # Save masks
    print("\nSaving masks...")
    mask_file = output_dir / f"{filename_base}_masks.tif"
    imwrite(mask_file, masks)
    print(f"  Saved: {mask_file.name}")
    
    # Save flows if available
    if flows is not None:
        flows_file = output_dir / f"{filename_base}_flows.npz"
        np.savez_compressed(flows_file, dP=flows[0], cellprob=flows[1], p=flows[2])
        print(f"  Saved: {flows_file.name}")
        
        cellprob_file = output_dir / f"{filename_base}_cellprob.tif"
        imwrite(cellprob_file, flows[1].astype(np.float32))
        print(f"  Saved: {cellprob_file.name}")
    
    # Save styles if available
    if styles is not None:
        styles_file = output_dir / f"{filename_base}_styles.npy"
        np.save(styles_file, styles)
        print(f"  Saved: {styles_file.name}")
    
    # Save diams
    diams_file = output_dir / f"{filename_base}_diams.npy"
    np.save(diams_file, diams)
    print(f"  Saved: {diams_file.name}")
    
    # Save features
    print("\nSaving features...")
    num_channels = features.shape[1] - 4
    
    column_names = ['x_min', 'x_max', 'y_min', 'y_max']
    column_names.extend([name for name in channel_names])
    
    df = pd.DataFrame(features, columns=column_names)
    df.insert(0, 'label', range(1, len(df) + 1))
    
    csv_file = output_dir / f"{filename_base}_features.csv"
    df.to_csv(csv_file, index=False)
    print(f"  Saved: {csv_file.name} ({len(df)} cells)")
    
    npy_file = output_dir / f"{filename_base}_features.npy"
    np.save(npy_file, features)
    print(f"  Saved: {npy_file.name}")
    
    # Save metadata
    metadata_file = output_dir / f"{filename_base}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {metadata_file.name}")


def segment_tiff(input_path, output_dir, model_type='nuclei', diameter=17,
                 channels=[0, 1], flow_threshold=0.4, cellprob_threshold=0.0,
                 normalize_blocksize=None, save_intermediate=True,
                 available_memory_gb=None, gpu_id=0):
    """
    Segment a TIFF image with automatic tiling if needed.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"Using GPU device: {gpu_id}")
    
    # Load image
    print(f"\nLoading image: {input_path.name}")
    img = imread(input_path)
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    
    # Get metadata
    channel_names = get_ome_channel_names(input_path)
    resolution = get_ome_resolution(input_path)
    
    if channel_names:
        print(f"Channel names: {channel_names}")
    
    if resolution:
        print(f"Resolution: {resolution['x_resolution']:.4f} {resolution['unit']}/pixel")
        diameter_um = diameter * resolution['x_resolution']
        print(f"Diameter: {diameter} pixels = {diameter_um:.2f} {resolution['unit']}")
    
    # Determine image shape
    if img.ndim == 3 and img.shape[0] <= 10:  # (C, H, W)
        img_shape = img.shape
        height, width = img.shape[1], img.shape[2]
        num_channels = img.shape[0]
    elif img.ndim == 3:  # (H, W, C)
        img_shape = img.shape
        height, width = img.shape[0], img.shape[1]
        num_channels = img.shape[2]
    else:  # 2D
        img_shape = img.shape
        height, width = img.shape
        num_channels = 1
    
    if channel_names is None:
        channel_names = [f'channel_{i}' for i in range(num_channels)]
    
    # Calculate tile configuration
    tile_size, overlap = calculate_tile_size(img_shape, diameter, available_memory_gb)
    
    # Initialize model
    print(f"\nLoading Cellpose model: {model_type}")
    model = models.Cellpose(gpu=True, model_type=model_type)
    
    normalize_params = {"tile_norm_blocksize": normalize_blocksize} if normalize_blocksize else None
    
    # Process based on whether tiling is needed
    if tile_size is None:
        print("\nProcessing entire image (no tiling)")
        
        masks, flows, styles, diams = model.eval(
            img,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            normalize=normalize_params
        )
        
        num_cells = len(np.unique(masks)) - 1
        print(f"Detected {num_cells} cells")
        
        used_tiling = False
        num_tiles = 1
        
    else:
        # Tiled processing
        if save_intermediate:
            intermediate_dir = output_dir / 'intermediate_tiles'
            intermediate_dir.mkdir(exist_ok=True)
        
        tiles = generate_tiles(height, width, tile_size, overlap)
        print(f"\nProcessing {len(tiles)} tiles...")
        
        tile_results = []
        
        for tile_idx, (y_start, y_end, x_start, x_end, idx) in enumerate(tiles):
            print(f"\nTile {tile_idx + 1}/{len(tiles)}: [{y_start}:{y_end}, {x_start}:{x_end}]")
            
            # Extract tile
            if img.ndim == 3 and img.shape[0] <= 10:  # (C, H, W)
                img_tile = img[:, y_start:y_end, x_start:x_end]
            else:
                img_tile = img[y_start:y_end, x_start:x_end]
            
            # Segment tile
            masks_tile, flows_tile, styles_tile, diams_tile = model.eval(
                img_tile,
                diameter=diameter,
                channels=channels,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                normalize=normalize_params
            )
            
            num_cells_tile = len(np.unique(masks_tile)) - 1
            print(f"  Detected {num_cells_tile} cells in tile")
            
            result = {
                'masks': masks_tile,
                'flows': flows_tile,
                'styles': styles_tile,
                'diams': diams_tile,
                'coords': (y_start, y_end, x_start, x_end),
                'tile_idx': idx
            }
            tile_results.append(result)
            
            # Save intermediate
            if save_intermediate:
                tile_file = intermediate_dir / f"tile_{idx:04d}.npz"
                np.savez_compressed(
                    tile_file,
                    masks=masks_tile,
                    cellprob=flows_tile[1],
                    coords=np.array([y_start, y_end, x_start, x_end]),
                    diams=diams_tile
                )
                print(f"  Saved intermediate: {tile_file.name}")
        
        # Merge tiles
        print("\nStitching tiles together...")
        masks = merge_tiles(tile_results, tiles, (height, width), overlap)
        
        diams = np.mean([r['diams'] for r in tile_results])
        flows = None
        styles = None
        
        num_cells = len(np.unique(masks)) - 1
        print(f"\nFinal cell count: {num_cells}")
        
        used_tiling = True
        num_tiles = len(tiles)
    
    # Extract features
    print("\nExtracting cell features...")
    features = extract_cell_features(img, masks)
    
    # Prepare metadata
    metadata = {
        'input_file': str(input_path),
        'timestamp': datetime.now().isoformat(),
        'image_shape': list(img.shape),
        'num_cells': int(num_cells),
        'model_type': model_type,
        'diameter': int(diameter),
        'channels': channels,
        'flow_threshold': float(flow_threshold),
        'cellprob_threshold': float(cellprob_threshold),
        'used_tiling': used_tiling,
        'tile_size': int(tile_size) if tile_size else None,
        'overlap': int(overlap) if overlap else None,
        'num_tiles': num_tiles,
        'channel_names': channel_names,
        'resolution': resolution
    }
    
    # Save outputs
    filename_base = input_path.stem
    save_outputs(masks, flows, styles, diams, features, output_dir, 
                 filename_base, channel_names, metadata)
    
    print(f"\n{'='*60}")
    print(f"Segmentation complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total cells: {num_cells}")
    print(f"{'='*60}")
    
    return masks, features, diams


def main():
    parser = argparse.ArgumentParser(
        description='Segment TIFF image with Cellpose (automatic tiling for large images)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('input_tiff', type=str,
                        help='Input TIFF file')
    parser.add_argument('output_dir', type=str,
                        help='Output directory')
    parser.add_argument('--model', type=str, default='nuclei',
                        choices=['cyto', 'nuclei', 'cyto2', 'cyto3'],
                        help='Cellpose model type')
    parser.add_argument('--diameter', type=int, default=17,
                        help='Expected cell diameter in pixels')
    parser.add_argument('--channels', type=int, nargs=2, default=[0, 1],
                        help='[cytoplasm_channel nucleus_channel]')
    parser.add_argument('--flow-threshold', type=float, default=0.4,
                        help='Flow error threshold')
    parser.add_argument('--cellprob-threshold', type=float, default=0.0,
                        help='Cell probability threshold')
    parser.add_argument('--normalize-blocksize', type=int, default=None,
                        help='Tile normalization block size')
    parser.add_argument('--no-intermediate', action='store_true',
                        help='Skip saving intermediate tile results')
    parser.add_argument('--available-memory', type=float, default=None,
                        help='Available GPU memory in GB (if not specified, auto-detect)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    
    args = parser.parse_args()
    
    segment_tiff(
        input_path=args.input_tiff,
        output_dir=args.output_dir,
        model_type=args.model,
        diameter=args.diameter,
        channels=args.channels,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        normalize_blocksize=args.normalize_blocksize,
        save_intermediate=not args.no_intermediate,
        available_memory_gb=args.available_memory,
        gpu_id=args.gpu
    )


if __name__ == '__main__':
    main()
