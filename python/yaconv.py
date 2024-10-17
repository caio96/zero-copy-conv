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

    # print("\nAll image ", images.shape)
    # print(np.squeeze(images))
    #
    # print("\nAll filters ", filters.shape)
    # print(np.squeeze(filters))

    for fh in range(FH):

        # Calculate height slice of size OH and handle edge cases
        height_offset = fh - PH
        height_start = max(0, height_offset)
        height_end = min(H, height_offset + OH)
        height_slice = height_end - height_start

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

            # Image is N,H,W,C
            # Select image slice of size N,OH,FW,C
            image_slice = images[:, height_start:height_end, width_start:width_end, :]

            # Flattened filter: 1,FW,C,M -> FWxC,M
            flattened_filter = np.reshape(filter_slice, (-1, filter_slice.shape[-1]))
            # Flattened image: N,OH,FW,C -> NxOH,FWxC
            flattened_image = np.reshape(
                image_slice, (image_slice.shape[0] * image_slice.shape[1], -1)
            )

            # print("\nImage ", flattened_image.shape)
            # print(np.squeeze(flattened_image))
            # print("\nFilter ", flattened_filter.shape)
            # print(np.squeeze(flattened_filter))

            # Results: NxOH,M -> N,OH,M
            result = np.reshape(
                np.matmul(flattened_image, flattened_filter),
                (N, image_slice.shape[1], filter_slice.shape[-1]),
            )

            # print("\nResult ", result.shape)
            # print(np.squeeze(result))

            # Output is N,OH,OW,M
            # Select output slice of size N,OH,1,M and handle edge cases
            if height_offset < 0:
                output_slice = outputs[:, -height_offset : -height_offset + height_slice, ow, :]
            else:
                output_slice = outputs[:, :height_slice, ow, :]

            # Place the result in the output matrix
            output_slice += result

            # print("\nOutput Slice ", output_slice.shape)
            # print(np.squeeze(output_slice))

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

    # # Create images and filters with values from 1 to their size
    # images = np.arange(1, args.N * args.H * args.W * args.C + 1).reshape(
    #     args.N, args.H, args.W, args.C
    # )
    # filters = np.arange(1, args.FH * args.FW * args.C * args.M + 1).reshape(
    #     args.FH, args.FW, args.C, args.M
    # )

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
