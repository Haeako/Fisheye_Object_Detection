import tensorrt as trt
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    builder = trt.Builder(TRT_LOGGER)
    network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    success = parser.parse_from_file(onnx_file_path)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        raise RuntimeError("Failed to parse ONNX")

    print("Parsing ONNX model...")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB

    # === Optimization Profile for dynamic input ===
    profile = builder.create_optimization_profile()

    input_name = network.get_input(0).name
    input_shape = network.get_input(0).shape  # e.g. [-1, 3, 640, 640]
    print(f"{input_name} has input shape: {input_shape}")

    profile.set_shape(
        "images",
        min = (1, 3, 640, 640),
        opt = (1, 3, 640, 640),
        max = (4, 3, 640, 640)
    )
    profile.set_shape(
        "orig_target_sizes",
        min = (1, 2),
        opt = (1, 2),
        max = (4,2)
    )
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"Engine saved to: {engine_file_path}")
    return engine_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='onnx2trt',
                    description='Convert ONNX to TensorRT engine')
    parser.add_argument('onnx_path')           
    parser.add_argument('-o', '--output')
    args = parser.parse_args()
    onnx_model_path = args.onnx_path
    engine_file_path = args.output
    build_engine(onnx_model_path, engine_file_path)
