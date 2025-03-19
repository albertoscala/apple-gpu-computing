# Apple GPU Computing with Metal

This repository contains an example project demonstrating image processing parallelized on Apple Silicon GPUs using Metal. The project applies various image transformations, showcasing the power of Metal for high-performance GPU computing on macOS.

## Features
- Image processing accelerated with Metal
- Parallel execution on Apple Silicon GPUs
- Example implementations for common image operations
- Comparison of CPU vs. GPU performance

## Prerequisites
- macOS with Apple Silicon (M1/M2/M3 or later)
- Xcode with Metal support
- C++ and Metal programming knowledge recommended

## Installation
1. Clone the repository:

git clone https://github.com/yourusername/apple-gpu-computing.git
cd apple-gpu-computing


2. Open the project in Xcode.
3. Build and run on a Mac with an Apple GPU.

## Usage
- The test/ directory contains sample images used for processing.
- The results/ directory stores output images after processing.
- Modify main.mm or the Metal shader files to experiment with different image processing techniques.

## Performance Comparison

The project includes CPU and GPU implementations for image processing tasks.

## File Structure

apple-gpu-computing/
│── src/                  # Source code for CPU and GPU processing  
│── shaders/              # Metal shader files  
│── test/                 # Input images for testing  
│── results/              # Output images after processing  
│── README.md             # Project documentation  
│── run_benchmark.sh      # Benchmark script  
