Cellpose TIFF Segmentation Pipeline
A high-performance, GPU-accelerated pipeline for segmenting large microscopy images using Cellpose. Features intelligent tiling for memory-constrained processing and SLURM-based batch processing for multi-sample workflows.
Features

🧠 Intelligent Memory Management: Automatically tiles large images that don't fit in GPU memory
🔬 OME-TIFF Support: Extracts channel names and resolution metadata from OME-TIFF files
🚀 Batch Processing: SLURM-based pipeline for processing multiple samples in parallel
📊 Comprehensive Outputs: Generates masks, features (CSV/NPY), cell probabilities, and metadata
🎯 Overlap Handling: IoU-based duplicate detection when merging tiled segmentations
💾 Flexible Input: Works with standard TIFFs, multi-channel TIFFs, and OME-TIFFs

Pipeline Architecture
cellpose2_pipeline.s (Main Controller)
    ↓
    Submits array job →
        ↓
    cellpose2_segmentation_controller.s (Per-sample jobs)
        ↓
        Runs →
            ↓
        cellpose2_segmentation.py (Core segmentation)
Installation
Requirements

Python 3.8+
CUDA-capable GPU
Cellpose 2.x
SLURM workload manager (for batch processing)
