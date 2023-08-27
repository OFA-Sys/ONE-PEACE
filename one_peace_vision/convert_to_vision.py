import pickle as pkl
import torch
import sys


if __name__ == '__main__':
    input = sys.argv[1]
    obj = torch.load(input, map_location="cpu")['model']

    newmodel = {}
    obj.pop("image_proj.weight")
    obj.pop("image_proj.bias")
    obj.pop("encoder_wrapper.fusion_model.image_layer_norm.weight")
    obj.pop("encoder_wrapper.fusion_model.image_layer_norm.bias")
    for k in list(obj.keys()):
        old_k = k
        if "text" in k or "audio" in k or "decoder" in k or "mask" in k or "logit_scale" in k or "version" in k:
            obj.pop(k)
            continue
        k = k.replace("encoder_wrapper", "encoder")
        k = k.replace("fusion_model.", "")
        if "image_adapter" in k:
            k = k.replace("encoder.", "")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach()

    res = {"model": newmodel, "__author__": "ofa", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())
