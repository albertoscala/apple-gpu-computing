//
//  main.cpp
//  apple-gpu-computing
//
//  Created by Alberto Scala on 17/03/25.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <chrono>

// Metal config
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
using namespace std;

struct Pixel {
    unsigned char r, g, b;
};

struct ImageSize {
    unsigned int width, height;
};

tuple<int, int, vector<struct Pixel>> read_image(string image_path) {
    tuple<int, int, vector<struct Pixel>> image_data;
    
    ifstream image(image_path, ios::binary);

    string format;
    int width, height, max_color;

    // Read PPM header
    image >> format >> width >> height >> max_color;
    image.ignore(1);

    if (format != "P6") {
        cerr << "Error: Format not supported: " << format << endl;
    }
    
    vector<struct Pixel> pixels(width * height);

    // Read pixel data
    image.read(reinterpret_cast<char*>(pixels.data()), width * height * 3);

    cout << "Pixels in the photo: " << pixels.size() << endl;

    image_data = make_tuple(width, height, pixels);

    return image_data;
}

// Adding 3 layer or padding
vector<struct Pixel> add_padding(int width, int height, vector<struct Pixel> image) {
    vector<struct Pixel> new_image = vector<struct Pixel>((width + 6) * (height + 6));

    // Copy the original image to the new image
    for (auto i = 0; i < height; i++) {
        for (auto j = 0; j < width; j++) {
            new_image[(i + 3) * (width + 6) + (j + 3)] = image[i * width + j];
        }
    }

    // Add padding to the top and bottom
    for (auto i = 0; i < 3; i++) {
        for (auto j = 0; j < width; j++) {
            new_image[i * (width + 6) + (j + 3)] = image[j];
            new_image[(height + 3 + i) * (width + 6) + (j + 3)] = image[(height - 1) * width + j];
        }
    }

    // Add padding to the left and right
    for (auto i = 0; i < height + 6; i++) {
        for (auto j = 0; j < 3; j++) {
            new_image[i * (width + 6) + j] = new_image[i * (width + 6) + 3];
            new_image[i * (width + 6) + (width + 3 + j)] = new_image[i * (width + 6) + (width + 2)];
        }
    }

    return new_image;
}

vector<struct Pixel> blur_image(int width, int height, vector<struct Pixel> image) {
    vector<struct Pixel> blurred_image((width - 6) * (height - 6));

    // Define the Gaussian kernel (7x7)
    float kernel[7][7] = {
        {1, 1, 1, 1, 1, 1, 1},
        {1, 2, 2, 2, 2, 2, 1},
        {1, 2, 3, 3, 3, 2, 1},
        {1, 2, 3, 4, 3, 2, 1},
        {1, 2, 3, 3, 3, 2, 1},
        {1, 2, 2, 2, 2, 2, 1},
        {1, 1, 1, 1, 1, 1, 1}
    };

    // Normalize the kernel
    float kernel_sum = 82.0;

    for (int i = 3; i < height - 3; i++) {
        for (int j = 3; j < width - 3; j++) {
            float r = 0, g = 0, b = 0;

            // Apply the kernel to the current position
            for (int ki = -3; ki <= 3; ki++) {
                for (int kj = -3; kj <= 3; kj++) {
                    struct Pixel pixel = image[(i + ki) * width + (j + kj)];
                    float kernel_value = kernel[ki + 3][kj + 3];

                    r += pixel.r * kernel_value;
                    g += pixel.g * kernel_value;
                    b += pixel.b * kernel_value;
                }
            }

            // Normalize the result
            r /= kernel_sum;
            g /= kernel_sum;
            b /= kernel_sum;

            // Assign the blurred pixel to the new image
            blurred_image[(i - 3) * (width - 6) + (j - 3)] = {static_cast<unsigned char>(r), static_cast<unsigned char>(g), static_cast<unsigned char>(b)};
        }
    }

    return blurred_image;
}

void write_image(string image_path, int width, int height, vector<struct Pixel> image) {
    ofstream image_file(image_path, ios::binary);

    image_file << "P6\n" << width << " " << height << "\n255\n";
    image_file.write(reinterpret_cast<char*>(image.data()), width * height * 3);
}

void singlethread_blur(string image_path) {
    // Reading the image
    tuple<int, int, vector<struct Pixel>> image_data = read_image(image_path);

    int width = get<0>(image_data);
    int height = get<1>(image_data);
    vector<struct Pixel> image = get<2>(image_data);

    // Adding padding to the image
    image = add_padding(width, height, image);

    cout << "Pixels: " << image.size() << endl;

    auto start = chrono::high_resolution_clock::now();
    
    vector<struct Pixel> blurred_image = blur_image(width+6, height+6, image);

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    cout << "Time taken: " << duration.count() << " microseconds" << endl;
    
    // Writing the image
    write_image("results/lisa_upscaled_blurred_cpu.ppm", width, height, blurred_image);
}

void multithread_blur(string image_path) {
    // Reading the image
    tuple<int, int, vector<struct Pixel>> image_data = read_image(image_path);

    int width = get<0>(image_data);
    int height = get<1>(image_data);
    vector<struct Pixel> image = get<2>(image_data);
    
    // Adding padding to the image
    image = add_padding(width, height, image);

    cout << "Pixels: " << image.size() << endl;

    vector<struct Pixel> blurred_image(width * height);
    
    // METAL
    
    // Create device
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal not supported!" << std::endl;
        return;
    }

    // Utilities for the GPU
    MTL::CommandQueue* commandQueue = device->newCommandQueue();
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

    // Prepare buffers TO CHECK
    size_t imageSize = (width+6) * (height+6) * sizeof(Pixel);
    MTL::Buffer* inputBuffer = device->newBuffer(imageSize, MTL::ResourceStorageModeShared);
    memcpy(inputBuffer->contents(), image.data(), imageSize);

    size_t outputSize = width * height * sizeof(Pixel);
    MTL::Buffer* outputBuffer = device->newBuffer(outputSize, MTL::ResourceStorageModeShared);

    // Image size buffer
    ImageSize imageSizeStruct = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    MTL::Buffer* imageSizeBuffer = device->newBuffer(sizeof(ImageSize), MTL::ResourceStorageModeShared);
    memcpy(imageSizeBuffer->contents(), &imageSizeStruct, sizeof(ImageSize));

    // Load shader
    NS::Error* error = nullptr;
    MTL::Library* library = device->newDefaultLibrary();
    MTL::Function* function = library->newFunction(NS::String::string("gaussian_blur", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pipeline = device->newComputePipelineState(function, &error);

    // Set pipeline and buffers
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer, 0, 0);
    encoder->setBuffer(outputBuffer, 0, 1);
    encoder->setBuffer(imageSizeBuffer, 0, 2); // Pass ImageSize struct

    // Dispatch threads
    MTL::Size gridSize = MTL::Size(width * height, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(1, 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);

    encoder->endEncoding();
    
    auto start = chrono::high_resolution_clock::now();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    cout << "Time taken: " << duration.count() << " microseconds" << endl;
    
    // Retrieve result
    memcpy(blurred_image.data(), outputBuffer->contents(), outputSize);
    
    write_image("results/lisa_upscaled_blurred_gpu.ppm", width, height, blurred_image);

    // Cleanup
    inputBuffer->release();
    outputBuffer->release();
    imageSizeBuffer->release();
    encoder->release();
    commandBuffer->release();
    commandQueue->release();
    pipeline->release();
    function->release();
    library->release();
    device->release();
}

int main() {
    string image_path = "test/lisa_upscaled.ppm";

    singlethread_blur(image_path);

    multithread_blur(image_path);
    
    return 0;
}
