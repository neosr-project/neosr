import warnings
from copy import deepcopy
from os import path as osp
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

# from onnxconverter_common.float16 import convert_float_to_float16
# from onnxsim import simplify
from neosr.archs import build_network
from neosr.utils.options import parse_options

"""
TODO

def rep_rt4ksr(net):
    rep_model = net.eval()
    rep_state_dict = net.eval().state_dict()
    pretrained_state_dict = net.train().state_dict()

    for k, v in rep_state_dict.items():
        if "rep_conv.weight" in k:
            # merge conv1x1-conv3x3-conv1x1
            k0 = pretrained_state_dict[k.replace("rep", "expand")]
            k1 = pretrained_state_dict[k.replace("rep", "fea")]
            k2 = pretrained_state_dict[k.replace("rep", "reduce")]

            bias_str = k.replace("weight", "bias")
            b0 = pretrained_state_dict[bias_str.replace("rep", "expand")]
            b1 = pretrained_state_dict[bias_str.replace("rep", "fea")]
            b2 = pretrained_state_dict[bias_str.replace("rep", "reduce")]

            mid_feats, n_feats = k0.shape[:2]

            # first step: remove the middle identity
            for i in range(mid_feats):
                k1[i, i, 1, 1] += 1.0

            # second step: merge the first 1x1 convolution and the next 3x3 convolution
            merged_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
            merged_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, mid_feats, 3, 3).cuda()
            merged_b0b1 = F.conv2d(input=merged_b0b1, weight=k1, bias=b1)

            # third step: merge the remain 1x1 convolution
            merged_k0k1k2 = F.conv2d(input=merged_k0k1.permute(1, 0, 2, 3), weight=k2).permute(1, 0, 2, 3)
            merged_b0b1b2 = F.conv2d(input=merged_b0b1, weight=k2, bias=b2).view(-1)

            # last step: remove the global identity
            for i in range(n_feats):
                merged_k0k1k2[i, i, 1, 1] += 1.0

            # save merged weights and biases in rep state dict
            rep_state_dict[k] = merged_k0k1k2.float()
            rep_state_dict[bias_str] = merged_b0b1b2.float()

        elif "rep_conv.bias" in k:
            pass

        elif k in pretrained_state_dict.keys():
            rep_state_dict[k] = pretrained_state_dict[k]

    rep_model.load_state_dict(rep_state_dict, strict=True)
    return rep_model
"""


def load_net():
    # build_network
    print(f"\n-------- Attempting to build network [{args.network}].")
    if args.network is None:
        msg = "Please select a network using the -net option"
        raise ValueError(msg)
    net_opt = {"type": args.network}

    if args.network == "omnisr":
        net_opt["upsampling"] = args.scale
        net_opt["window_size"] = args.window

    if args.window:
        net_opt["window_size"] = args.window

    load_net = torch.load(
        args.input, map_location=torch.device("cuda"), weights_only=True
    )
    # find parameter key
    print("-------- Finding parameter key...")
    param_key: str | None = None
    try:
        if "params-ema" in load_net:
            param_key = "params-ema"
        elif "params" in load_net:
            param_key = "params"
        elif "params_ema" in load_net:
            param_key = "params_ema"
        load_net = load_net[param_key]
    except:
        pass

    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith("module."):
            load_net[k[7:]] = v
            load_net.pop(k)

    net = build_network(net_opt)
    # TODO
    """
    if args.network.startswith("rt4ksr"):
        net = rep_rt4ksr(net)
    else:
    """
    # load_network and send to device and set to eval mode
    net.load_state_dict(load_net, strict=True)  # type: ignore[reportAttributeAccessIssue,attr-defined]
    net.eval()

    # plainusr
    if args.network == "plainusr":
        try:
            for module in net.modules():
                if hasattr(module, "switch_to_deploy"):
                    module.switch_to_deploy(args.prune)
            print("-------- Reparametrization completed successfully.")
        except:
            pass

    net = net.to(device="cuda", non_blocking=True)  # type: ignore[reportAttributeAccessIssue,attr-defined]
    print(f"-------- Successfully loaded network [{args.network}].")
    torch.cuda.empty_cache()

    return net


def assert_verify(onnx_model, torch_model) -> None:
    if args.static is not None:
        dummy_input = torch.randn(1, *args.static, requires_grad=True)
    else:
        dummy_input = torch.randn(1, 3, 20, 20, requires_grad=True)
    # onnxruntime output prediction
    # NOTE: "CUDAExecutionProvider" errors if some nvidia libs
    # are not found, defaulting to cpu
    ort_session = onnxruntime.InferenceSession(
        onnx_model, providers=["CPUExecutionProvider"]
    )
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.detach().cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # torch outputs
    with torch.inference_mode():
        torch_outputs = torch_model(dummy_input)

    # final assert - default tolerance values - rtol=1e-03, atol=1e-05
    np.testing.assert_allclose(
        torch_outputs.detach().cpu().numpy(), ort_outs[0], rtol=0.01, atol=0.001
    )
    print("-------- Model successfully verified.")


def to_onnx() -> None:
    # error if network can't be converted
    net_error = ["craft", "ditn"]
    if args.network in net_error:
        msg = f"Network [{args.network}] cannot be converted to ONNX."
        raise RuntimeError(msg)

    # load network and send to device
    model = load_net()
    # set model to eval mode
    model.eval()

    # set static or dynamic
    if args.static is not None:
        dummy_input = torch.randn(1, *args.static, requires_grad=True)
    else:
        dummy_input = torch.randn(1, 3, 20, 20, requires_grad=True)

    # dict for dynamic axes
    if args.static is None:
        dyn_axes = {
            "dynamic_axes": {
                "input": {0: "batch_size", 2: "width", 3: "height"},
                "output": {0: "batch_size", 2: "width", 3: "height"},
            },
            "input_names": ["input"],
            "output_names": ["output"],
        }
    else:
        dyn_axes = {"input_names": ["input"], "output_names": ["output"]}

    # add _fp32 suffix to output str
    filename, extension = osp.splitext(args.output)  # noqa: PTH122
    output_fp32 = filename + "_fp32" + extension
    # begin conversion
    print("-------- Starting ONNX conversion (this can take a while)...")

    with torch.inference_mode(), torch.device("cpu"):
        # TODO: add param dynamo=True as a switch
        # py2.5 supports the verify=True flag now as well
        torch.onnx.export(
            model,
            dummy_input,
            output_fp32,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=False,
            **(dyn_axes or {}),  # type: ignore
        )

    print("-------- Conversion was successful. Verifying...")
    # verify onnx
    load_onnx = onnx.load(output_fp32)
    torch.cuda.empty_cache()
    onnx.checker.check_model(load_onnx)
    print(
        f"-------- Model successfully converted to ONNX format. Saved at: {output_fp32}."
    )
    # verify outputs
    if args.nocheck is False:
        assert_verify(output_fp32, model)

    if args.optimize:
        print("-------- Running ONNX optimization...")
        filename, extension = osp.splitext(args.output)  # noqa: PTH122
        output_optimized = filename + "_fp32_optimized" + extension
        session_opt = onnxruntime.SessionOptions()
        # ENABLE_ALL can cause compatibility issues, leaving EXTENDED as default
        session_opt.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        session_opt.optimized_model_filepath = output_optimized
        # save
        onnxruntime.InferenceSession(output_fp32, session_opt)
        # verify
        onnx.checker.check_model(onnx.load(output_optimized))
        print(f"-------- Model successfully optimized. Saved at: {output_optimized}")

    if args.fp16:
        print(
            "-------- fp16 conversion has been temporarily disabled due to onnxconverter-common not being maintained."
        )
        """
        print("-------- Converting to fp16...")
        output_fp16 = filename + "_fp16" + extension
        # convert to fp16
        if args.optimize:
            to_fp16 = convert_float_to_float16(onnx.load(output_optimized))  # type: ignore[reportPossiblyUnboundVariable]
        else:
            to_fp16 = convert_float_to_float16(load_onnx)
        # save
        onnx.save(to_fp16, output_fp16)
        # verify
        onnx.checker.check_model(onnx.load(output_fp16))
        print(
            f"-------- Model successfully converted to half-precision. Saved at: {output_fp16}."
        )
        """

    if args.fulloptimization:
        msg = "ONNXSimplify has been temporarily disabled."
        raise ValueError(msg)
        """
        # error if network can't run through onnxsim
        opt_error = ["omnisr"]
        if args.network in opt_error:
            msg = f"Network [{args.network}] doesnt support full optimization."
            raise RuntimeError(msg)

        print("-------- Running full optimization (this can take a while)...")
        output_fp32_fulloptimized = filename + "_fp32_fullyoptimized" + extension
        output_fp16_fulloptimized = filename + "_fp16_fullyoptimized" + extension

        # run onnxsim
        if args.optimize:
            simplified, check = simplify(onnx.load(output_optimized))
        elif args.fp16:
            simplified, check = simplify(onnx.load(output_fp16))
        else:
            simplified, check = simplify(load_onnx)
        assert check, "Couldn't validate ONNX model."

        # save and verify
        if args.fp16:
            onnx.save(simplified, output_fp16_fulloptimized)
            onnx.checker.check_model(onnx.load(output_fp16_fulloptimized))
        else:
            onnx.save(simplified, output_fp32_fulloptimized)
            onnx.checker.check_model(onnx.load(output_fp32_fulloptimized))

        print(
            f"-------- Model successfully optimized. Saved at: {output_fp32_fulloptimized}\n"
        )
        """


if __name__ == "__main__":
    torch.set_default_device("cuda")
    warnings.filterwarnings("ignore", category=UserWarning)
    root_path = Path(Path(__file__) / osp.pardir).resolve()
    __, args = parse_options(str(root_path))
    to_onnx()
