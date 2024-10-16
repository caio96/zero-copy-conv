#!/usr/bin/env python
import argparse

import numpy as np
import torch
import torch.nn.functional as F


def pytorch_conv2d(images, filters, padding):
    # (N, H, W, C) -> (N, C, H, W)
    # (FH, FW, C, M) -> (M, C, FH, FW)
    images_torch = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    filters_torch = torch.tensor(filters, dtype=torch.float32).permute(3, 2, 0, 1)

    # Perform 2D convolution
    outputs_torch = F.conv2d(images_torch, filters_torch, stride=1, padding=padding)

    # (N, M, HO, WO) -> (N, HO, WO, M)
    return outputs_torch.permute(0, 2, 3, 1).numpy()


def yaconv_conv2d(images, filters, padding):
    N, H, W, C = images.shape
    FH, FW, _, M = filters.shape
    PH, PW = padding
    OH, OW = H + 2 * PH - FH + 1, W + 2 * PW - FW + 1

    outputs = np.zeros((N, OH, OW, M))

    for n in range(N):
        # H,W,C -> W,C,H
        single_image = np.transpose(images[n], (1, 2, 0))
        # OH,OW,M
        single_output = outputs[n]

        for fh in range(FH):
            for ow in range(OW):
                # Calculate width slice of size FW and handle edge cases
                ow_offset = ow - PW
                width_start = max(0, ow_offset)
                width_end = min(W, ow_offset + FW)
                filter_width_slice = width_end - width_start

                # Filter is FH,FW,C,M
                # Select filter slice of size 1,FW,C,M
                if ow_offset < 0:
                    filter_slice = filters[fh, -ow_offset : -ow_offset + filter_width_slice, :, :]
                else:
                    filter_slice = filters[fh, :filter_width_slice, :, :]

                # Calculate height slice of size OH and handle edge cases
                height_offset = fh - PH
                height_start = max(0, height_offset)
                height_end = min(H, height_offset + OH)
                height_slice = height_end - height_start

                # Single image is W,C,H
                # Select image slice of size FW,C,OH
                image_slice = single_image[width_start:width_end, :, height_start:height_end]

                # Flattened filter: 1,FW,C,M -> M,FWxC
                flattened_filter = np.transpose(
                    np.reshape(filter_slice, (-1, filter_slice.shape[-1]))
                )
                # Flattened image: FW,C,OH -> FWxC,OH
                flattened_image = np.reshape(image_slice, (-1, image_slice.shape[-1]))
                # Results is M,OH, then, transposed to OH,M
                result = np.transpose(np.matmul(flattened_filter, flattened_image))

                # Single output is OH,OW,M
                # Select output slice of size OH,1,M and handle edge cases
                if height_offset < 0:
                    output_slice = single_output[
                        -height_offset : -height_offset + height_slice, ow, :
                    ]
                else:
                    output_slice = single_output[:height_slice, ow, :]

                # Place the result in the output matrix
                output_slice += result

    return outputs


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Perform 2D convolution with a simplified version of Yaconv."
    )

    # Image parameters
    parser.add_argument("--N", type=int, default=2, help="Number of images (batch size)")
    parser.add_argument("--H", type=int, default=56, help="Height of images")
    parser.add_argument("--W", type=int, default=56, help="Width of images")
    parser.add_argument("--C", type=int, default=3, help="Number of channels in images")

    # Filter parameters
    parser.add_argument("--FH", type=int, default=3, help="Height of filters")
    parser.add_argument("--FW", type=int, default=3, help="Width of filters")
    parser.add_argument("--M", type=int, default=8, help="Number of filters (output channels)")

    # Padding
    parser.add_argument("--PH", type=int, default=1, help="Padding height")
    parser.add_argument("--PW", type=int, default=1, help="Padding width")

    # Seed for reproducibility
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    # Create random images and filters
    images = np.random.rand(args.N, args.H, args.W, args.C)
    filters = np.random.rand(args.FH, args.FW, args.C, args.M)

    # Print configurations (optional, can be commented out)
    print(f"Images: {images.shape}")
    print(f"Filters: {filters.shape}")

    # Perform convolution with both implementations
    pytorch_output = pytorch_conv2d(images, filters, padding=(args.PH, args.PW))
    yaconv_output = yaconv_conv2d(images, filters, padding=(args.PH, args.PW))

    # Compare results
    if np.allclose(pytorch_output, yaconv_output):
        print("✅ Yaconv and Torch match")
    else:
        print("❌ Yaconv and Torch differ")
