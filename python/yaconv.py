#!/usr/bin/env python
import numpy as np
import torch
import torch.nn.functional as F

def pytorch_conv2d(images, filters, padding):
    # Convert NumPy arrays to PyTorch tensors
    images_torch = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # N, H, W, C -> N, C, H, W
    filters_torch = torch.tensor(filters, dtype=torch.float32).permute(3, 2, 0, 1)  # FH, FW, C, M -> M, C, FH, FW

    # Perform 2D convolution
    outputs = F.conv2d(images_torch, filters_torch, stride=1, padding=padding)

    # Convert the output tensor back to NumPy array and permute to match the original output shape
    return outputs.permute(0, 2, 3, 1).numpy()  # N, C_out, H_out, W_out -> N, H_out, W_out, C_out

def yaconv_conv2d(images, filters, padding):
    N,H,W,C = images.shape
    FH,FW,C,M = filters.shape
    PH,PW = padding
    OH = H + 2 * PH - FH + 1;
    OW = W + 2 * PW - FW + 1;

    outputs = np.zeros((N,OH,OW,M))
    for n in range(N):
        # H,W,C -> W,C,H
        single_image = np.transpose(images[n], (1,2,0))
        single_output = outputs[n]

        for fh in range(FH):
            for ow in range(OW):
                # Slice of FW size of image W dimension
                ow_padding = ow - PW;
                image_start = ow_padding;
                image_end = min(W, image_start + FW);
                if (ow_padding < 0):
                  image_start = 0;
                w_slice = image_end - image_start;
                assert(w_slice > 0)

                if (ow_padding < 0):
                  filter = filters[fh, -1 * ow_padding: -1 * ow_padding+w_slice, :, :]
                  a = np.transpose(np.reshape(filter, ((FW+ow_padding)*C, M)))
                else:
                  filter = filters[fh, :w_slice, :, :]
                  # a is (M, FW*C)
                  a = np.transpose(np.reshape(filter, (w_slice*C, M)))

                oh_padding = fh - PH
                height_start = oh_padding
                height_end = min(H, height_start+OH)
                if (height_start < 0):
                  height_start = 0
                h_slice = height_end - height_start

                single_image_slice = single_image[image_start:image_end, :, height_start:height_end]
                b = np.reshape(single_image_slice, (single_image_slice.shape[0]*single_image_slice.shape[1],single_image_slice.shape[2]))

                # b is (FW*C, OH)
                # print("\nFilter ", a.shape)
                # print(np.squeeze(a))
                # print("\nImage ", b.shape)
                # print(np.squeeze(b))

                c = np.transpose(np.matmul(a,b))

                # print("\nC ", c.shape)
                # print(np.squeeze(c))
                # print(height_start, height_end)

                if (oh_padding < 0):
                    single_output[-1 * oh_padding: -1 * oh_padding+h_slice, ow, :] += c
                else:
                    single_output[:h_slice, ow, :] += c

                # print("\nOutput ", single_output.shape)
                # print(np.squeeze(single_output))
    return outputs


if __name__ == "__main__":
    # N,H,W,C = 1,3,3,1
    # FH,FW,M = 2,2,1
    # PH, PW = 0,0
    # images = np.reshape(np.arange(1, N*H*W*C+1),(N,H,W,C))
    # filters = np.reshape(np.arange(1, FH*FW*C*M+1),(FH,FW,C,M))

    N,H,W,C = 2,16,16,3
    FH,FW,M = 3,3,8
    PH, PW = 1,1
    images = np.random.rand(N,H,W,C)
    filters = np.random.rand(FH,FW,C,M)

    # print("images\n", images)
    # print("\nfilters\n", filters)

    pytorch_output = pytorch_conv2d(images, filters, padding=(PH, PW))
    yaconv_output = yaconv_conv2d(images, filters, padding=(PH, PW))

    if np.allclose(pytorch_output, yaconv_output):
        print("✅ Yaconv and Torch match")
    else:
        print("❌ Yaconv and Torch differ")

