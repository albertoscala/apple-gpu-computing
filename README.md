# Apple GPU Computing with Metal

This repository contains an example project demonstrating image processing parallelized on Apple Silicon GPUs using Metal. The project applies gaussian blur, showcasing the power of Metal for high-performance GPU computing on macOS.

## Features
- Image processing accelerated with Metal
- Parallel execution on Apple Silicon GPUs
- Example implementation for gaussian blur
- Comparison of CPU vs. GPU performance

## Prerequisites
- macOS with Apple Silicon (M1/M2/M3 or later)
- Xcode with Metal support
- C++ and Metal programming knowledge recommended

## Installation
1. Clone the repository:

```bash
git clone https://github.com/albertoscala/apple-gpu-computing/
cd apple-gpu-computing
```

2. Open the project in Xcode.
3. Build and run on a Mac with an Apple GPU.

## Usage
- The `test/` directory contains sample images used for processing.
- The `results/` directory stores output images after processing.

## File Structure
```
apple-gpu-computing/
│── apple-gpu-computing.xcodeproj/    # Xcode project files and configurations  
│── apple-gpu-computing/              # Main source directory containing CPU and GPU code  
│   │── test/                         # Sample input images for testing the processing pipeline  
│   │── results/                      # Output images generated after GPU processing  
│   │── compute.metal                 # Metal shader file for GPU-accelerated image processing  
│   │── main.cpp                      # Main application logic for loading, processing, and saving images  
│── metal-cpp/                        # C++ bindings and utilities for interfacing with Metal  
│── LICENSE                           # License information for the project  
│── README.md                         # Project documentation and usage instructions  
```
