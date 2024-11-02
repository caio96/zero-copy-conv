#!/usr/bin/env python
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F


def pytorch_conv2d(images, filters, padding, stride, dilation):
    # (N, H, W, C) -> (N, C, H, W)
    # (FH, FW, C, M) -> (M, C, FH, FW)
    images_torch = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    filters_torch = torch.tensor(filters, dtype=torch.float32).permute(3, 2, 0, 1)

    # Perform 2D convolution
    outputs_torch = F.conv2d(
        images_torch, filters_torch, stride=stride, padding=padding, dilation=dilation
    )

    # (N, M, HO, WO) -> (N, HO, WO, M)
    return outputs_torch.permute(0, 2, 3, 1).numpy()


def yaconv_conv2d(images, filters, padding, stride, dilation):
    N, H, W, C = images.shape
    FH, FW, _, M = filters.shape
    PH, PW = padding
    SH, SW = stride
    DH, DW = dilation
    OH = math.floor((H + 2 * PH - DH * (FH - 1) - 1) / SH + 1)
    OW = math.floor((W + 2 * PW - DW * (FW - 1) - 1) / SW + 1)

    outputs = np.zeros((N, OH, OW, M))

    # print("\nAll image ", images.shape)
    # print(np.squeeze(images))
    # print("\nAll filters ", filters.shape)
    # print(np.squeeze(filters))
    # print(f"\nOutput shape: {outputs.shape}")

    for n in range(N):
        single_image = images[n, :, :, :]
        single_output = outputs[n, :, :, :]

        for fh in range(FH):
            # Calculate height slice of size OH and handle edge cases
            height_offset = fh * DH - PH
            if height_offset < 0:
                height_start = max(0, height_offset % SH)
            else:
                height_start = max(0, height_offset)
            height_end = min(H, height_offset + OH * SH)
            height_slice = math.ceil((height_end - height_start) / SH)

            # # print height variables
            # print(f"\nFilter Height: {fh} ----------------")
            # print(f"Height Offset: {height_offset}")
            # print(f"Height Slice: ({height_end} - {height_start}) / {SH} = {height_slice}")

            if height_slice <= 0:
                continue

            for ow in range(OW):
                # Calculate width slice of size FW and handle edge cases
                # ow_offset = ow - PW
                iw = ow * SW - PW
                if iw < 0:
                    width_start = max(0, iw % DW)
                else:
                    width_start = max(0, iw)
                width_end = min(W, iw + FW * DW)
                filter_width_slice = math.ceil((width_end - width_start) / DW)

                # # Print width variables
                # print(f"\nOutput Width: {ow} ----------------")
                # print(f"Width Offset: {iw}")
                # print(f"Width Slice: ({width_end} - {width_start}) / {DW} = {filter_width_slice}")

                if filter_width_slice <= 0:
                    continue

                # Filter is FH,FW,C,M
                # Select filter slice of size 1,FW,C,M
                if iw < 0:
                    filter_slice = filters[fh, -(iw // DW) : -(iw // DW) + filter_width_slice, :, :]
                else:
                    filter_slice = filters[fh, :filter_width_slice, :, :]

                # Image is H,W,C
                # Select image slice of size OH,FW,C
                image_slice = single_image[height_start:height_end:SH, width_start:width_end:DW, :]

                # Flattened filter: 1,FW,C,M -> FWxC,M
                flattened_filter = np.reshape(filter_slice, (-1, filter_slice.shape[-1]))
                # Flattened image: OH,FW,C -> OH,FWxC
                flattened_image = np.reshape(image_slice, (image_slice.shape[0], -1))

                # print("\nImage ", flattened_image.shape)
                # print(np.squeeze(flattened_image))
                # print("\nFilter ", flattened_filter.shape)
                # print(np.squeeze(flattened_filter))

                result = np.matmul(flattened_image, flattened_filter)

                # print("\nResult ", result.shape)
                # print(np.squeeze(result))

                # Output is N,OH,OW,M
                # Select output slice of size N,OH,1,M and handle edge cases
                if height_offset < 0:
                    output_slice = single_output[
                        -(height_offset // SH) : -(height_offset // SH) + height_slice, ow, :
                    ]
                else:
                    output_slice = single_output[:height_slice, ow, :]

                # Place the result in the output matrix
                output_slice += result

                # print("\nOutput Slice ", output_slice.shape)
                # print(np.squeeze(output_slice))
                # print("\nOutput", outputs.shape)
                # print(np.squeeze(outputs))

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

    # Stride
    parser.add_argument("--SH", type=int, default=1, help="Stride height")
    parser.add_argument("--SW", type=int, default=1, help="Stride width")

    # Dilation
    parser.add_argument("--DH", type=int, default=1, help="Dilation height")
    parser.add_argument("--DW", type=int, default=1, help="Dilation width")

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

    # # Print configurations
    # print(f"Images: {images.shape}")
    # print(f"Filters: {filters.shape}")

    # # Print inputs
    # print("Images:")
    # print(np.squeeze(images))
    # print("Filters:")
    # print(np.squeeze(filters))

    # Perform convolution with both implementations
    pytorch_output = pytorch_conv2d(
        images,
        filters,
        padding=(args.PH, args.PW),
        stride=(args.SH, args.SW),
        dilation=(args.DH, args.DW),
    )
    yaconv_output = yaconv_conv2d(
        images,
        filters,
        padding=(args.PH, args.PW),
        stride=(args.SH, args.SW),
        dilation=(args.DH, args.DW),
    )

    # Compare results
    if np.allclose(pytorch_output, yaconv_output):
        print("✅ Yaconv and Torch match")
    else:
        print("❌ Yaconv and Torch differ")
        # print(f"Yaconv output:\n {np.squeeze(yaconv_output)}")
        # print(f"Torch output:\n {np.squeeze(pytorch_output)}")
