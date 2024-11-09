import functools
import os
import pickle
from math import floor

import timm
import torch
from PIL import Image
from tqdm import tqdm


class VerboseExecution(torch.nn.Module):
    def __init__(self, model, model_name, dims, input_size):
        super().__init__()
        self.model = model
        self.dims = dims
        self.input_size = input_size

        # Register a hook for each layer
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer.register_forward_hook(self.save_shapes_hook(name, model_name))

        with torch.no_grad():
            self.forward(torch.rand(input_size))

    def save_shapes_hook(self, module_name, model_name):
        def fn(module, inputs, outputs):

            inp = (1, *inputs[0].shape[1:])
            out = (1, *outputs[0].shape)

            # hash key
            key = f"{','.join([str(x) for x in inp])}_"  # input shape
            key += f"{','.join([str(x) for x in out])}_"  # output shape
            key += f"{','.join([str(x) for x in module.kernel_size])}-{','.join([str(x) for x in module._reversed_padding_repeated_twice])}-{','.join([str(x) for x in module.stride])}-{','.join([str(x) for x in module.dilation])}-{module.groups}-{module.transposed}_"  # kernel values
            key += f"{'True' if module.bias is not None else 'False'}"  # bias

            # populate dict
            self.dims.setdefault(key, []).append(model_name + "." + module_name)

        return fn

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    convs = dict()

    for m in tqdm(timm.list_models()):

        model = timm.create_model(m).eval()
        data_config = timm.data.resolve_model_data_config(model)

        model = VerboseExecution(model, m, convs, (1, *data_config["input_size"]))

        del model

    with open("conv.pkl", "wb") as handle:
        pickle.dump(convs, handle, protocol=pickle.HIGHEST_PROTOCOL)
