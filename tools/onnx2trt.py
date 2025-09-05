import tensorrt as trt
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path, use_fp16=True, use_best_optimization=False):
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
    
    # Enable FP16 precision
    if use_fp16:
        print("Enabling FP16 precision...")
        config.set_flag(trt.BuilderFlag.FP16)
        
        # Optional: Enable strict type constraints for better FP16 performance
        # config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    
    # Enable best optimization (slower build, better performance)
    if use_best_optimization:
        print("Enabling best optimization (this will take longer)...")
        # Set timing cache for better optimization
        config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        # Enable GPU fallback for unsupported layers
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        # More aggressive optimization
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    
    # === Optimization Profile for dynamic input ===
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    input_shape = network.get_input(0).shape  # e.g. [-1, 3, 1024, 1024]
    print(f"{input_name} has input shape: {input_shape}")
    
    profile.set_shape(
        "images",
        min=(1, 3, 1024, 1024),
        opt=(1, 3, 1024, 1024),
        max=(1, 3, 1024, 1024)
    )
    profile.set_shape(
        "orig_target_sizes",
        min=(1, 2),
        opt=(1, 2),
        max=(1, 2)
    )
    config.add_optimization_profile(profile)
    
    # Build the engine
    print("Building TensorRT engine... This may take a while.")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"Engine saved to: {engine_file_path}")
    
    return engine_file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='onnx2trt',
        description='Convert ONNX to TensorRT engine with FP16 support')
    parser.add_argument('onnx_path', help='Path to input ONNX model')
    parser.add_argument('-o', '--output', required=True, help='Path for output TensorRT engine')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 instead of FP16 (default: FP16)')
    parser.add_argument('--best', action='store_true', help='Enable best optimization (slower build, better runtime performance)')
    
    args = parser.parse_args()
    
    onnx_model_path = args.onnx_path
    engine_file_path = args.output
    use_fp16 = not args.fp32  # Default to FP16 unless --fp32 is specified
    use_best = args.best
    
    print(f"Converting {onnx_model_path} to TensorRT engine...")
    print(f"Precision: {'FP32' if not use_fp16 else 'FP16'}")
    print(f"Optimization: {'Best (slow build)' if use_best else 'Standard'}")
    
    build_engine(onnx_model_path, engine_file_path, use_fp16, use_best)