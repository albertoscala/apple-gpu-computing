//
//  compute.metal
//  apple-gpu-computing
//
//  Created by Alberto Scala on 18/03/25.
//

#include <metal_stdlib>
using namespace metal;


struct Pixel {
    uchar r, g, b;
};

// Pre-store 7x7 Gaussian kernel in constant GPU memory
constant float g_kernel[49] = {
    1, 1, 1, 1, 1, 1, 1,
    1, 2, 2, 2, 2, 2, 1,
    1, 2, 3, 3, 3, 2, 1,
    1, 2, 3, 4, 3, 2, 1,
    1, 2, 3, 3, 3, 2, 1,
    1, 2, 2, 2, 2, 2, 1,
    1, 1, 1, 1, 1, 1, 1
};

// Normalize kernel at compile-time
constant float kernel_sum = 82.0; // Sum of kernel values

// Struct to store width and height
struct ImageSize {
    uint width;
    uint height;
};

kernel void gaussian_blur(
                 device const Pixel* inputImage [[buffer(0)]],
                     device Pixel* outputImage [[buffer(1)]],
                 constant ImageSize* imageSize [[buffer(2)]],
                     uint id [[thread_position_in_grid]]
                 ) {
                     uint width = imageSize->width+6;
                         uint height = imageSize->height+6;

                        // Obtaining the coords for the pixel to work on
                         int x = id % width;
                         int y = id / width;

                         // The image data starts at (3, 3), so we start applying the kernel from here
                         int offsetX = 3;  // Offset due to padding
                         int offsetY = 3;  // Offset due to padding

                        
                        // IMPROVABLE
                         // Ensure we only process the actual image data area (ignoring the padding area)
                         if (x < offsetX || x >= width - offsetX || y < offsetY || y >= height - offsetY) {
                             return; // Skip the padding pixels
                         }

                         float r = 0, g = 0, b = 0;

                         for (int ky = -3; ky <= 3; ky++) {
                             for (int kx = -3; kx <= 3; kx++) {
                                 int index = (y + ky) * (width) + (x + kx);
                                 Pixel pixel = inputImage[index];
                                 float weight = g_kernel[(ky + 3) * 7 + (kx + 3)];

                                 r += pixel.r * weight;
                                 g += pixel.g * weight;
                                 b += pixel.b * weight;
                             }
                         }

                         // Normalize
                         r /= kernel_sum;
                         g /= kernel_sum;
                         b /= kernel_sum;

                         int outIndex = (y - 3) * (width - 6) + (x - 3);
                         outputImage[outIndex] = {static_cast<uchar>(r), static_cast<uchar>(g), static_cast<uchar>(b)};
    
}
