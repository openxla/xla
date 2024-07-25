class FlashAttentionBMMScalePaddingMaskSoftmaxBMM
    : public MultiHeadedAttentionTest {
 protected:
  const std::string  // NOLINT
  GetModuleFlash_Attention_Inference_BMM1_MoMask_Generation_Softmax_BMM2_HloString_BF16() {  // NOLINT
    const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[4,16,4,16]{3,2,1,0}, bf16[4,16,4,16]{3,2,1,0}, bf16[4,16,4,16]{3,2,1,0})->bf16[4,16,4,16]{3,2,1,0}}}, allow_spmd_sharding_propagation_to_parameters={true,true,true}, allow_spmd_sharding_propagation_to_output={true}

    clip.33 {
      Arg_2.36 = bf16[] parameter(2), metadata={op_name="jit(foo)/jit(main)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) in_layouts=(None, None, None) out_layouts=(None,) resource_env=None donated_invars=(False, False, False) name=clip keep_unused=False inline=False]"}
      broadcast.39 = bf16[4,16,4,16]{3,2,1,0} broadcast(Arg_2.36), dimensions={}, metadata={op_name="jit(foo)/jit(main)/jit(clip)/min" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=129}
      Arg_1.35 = bf16[] parameter(1), metadata={op_name="jit(foo)/jit(main)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) in_layouts=(None, None, None) out_layouts=(None,) resource_env=None donated_invars=(False, False, False) name=clip keep_unused=False inline=False]"}
      broadcast.37 = bf16[4,16,4,16]{3,2,1,0} broadcast(Arg_1.35), dimensions={}, metadata={op_name="jit(foo)/jit(main)/jit(clip)/max" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=129}
      Arg_0.34 = bf16[4,16,4,16]{3,2,1,0} parameter(0), metadata={op_name="jit(foo)/jit(main)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) in_layouts=(None, None, None) out_layouts=(None,) resource_env=None donated_invars=(False, False, False) name=clip keep_unused=False inline=False]"}
      maximum.38 = bf16[4,16,4,16]{3,2,1,0} maximum(broadcast.37, Arg_0.34), metadata={op_name="jit(foo)/jit(main)/jit(clip)/max" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=129}
      ROOT minimum.40 = bf16[4,16,4,16]{3,2,1,0} minimum(broadcast.39, maximum.38), metadata={op_name="jit(foo)/jit(main)/jit(clip)/min" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=129}
    } // clip.33


    ENTRY main.106 {
      Arg_0.1 = bf16[4,16,4,16]{3,2,1,0} parameter(0), metadata={op_name="quey"}
      constant.6 = bf16[] constant(1)
      broadcast.7 = bf16[4,16,4,16]{3,2,1,0} broadcast(constant.6), dimensions={}
      divide.8 = bf16[4,16,4,16]{3,2,1,0} divide(Arg_0.1, broadcast.7), metadata={op_name="jit(foo)/jit(main)/div" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=128}
      constant.5 = bf16[] constant(-448)
      constant.4 = bf16[] constant(448)
      call.17 = bf16[4,16,4,16]{3,2,1,0} call(divide.8, constant.5, constant.4), to_apply=clip.33
      convert.18 = f8e4m3fn[4,16,4,16]{3,2,1,0} convert(call.17), metadata={op_name="jit(foo)/jit(main)/convert_element_type[new_dtype=float8_e4m3fn weak_type=False]" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=130}
      convert.19 = bf16[4,16,4,16]{3,2,1,0} convert(convert.18), metadata={op_name="jit(foo)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=134}
      Arg_1.2 = bf16[4,16,4,16]{3,2,1,0} parameter(1), metadata={op_name="key"}
      divide.20 = bf16[4,16,4,16]{3,2,1,0} divide(Arg_1.2, broadcast.7), metadata={op_name="jit(foo)/jit(main)/div" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=128}
      call.29 = bf16[4,16,4,16]{3,2,1,0} call(divide.20, constant.5, constant.4), to_apply=clip.33
      convert.30 = f8e4m3fn[4,16,4,16]{3,2,1,0} convert(call.29), metadata={op_name="jit(foo)/jit(main)/convert_element_type[new_dtype=float8_e4m3fn weak_type=False]" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=130}
      convert.31 = bf16[4,16,4,16]{3,2,1,0} convert(convert.30), metadata={op_name="jit(foo)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=134}
      Arg_2.3 = bf16[4,16,4,16]{3,2,1,0} parameter(2), metadata={op_name="value"}
      divide.32 = bf16[4,16,4,16]{3,2,1,0} divide(Arg_2.3, broadcast.7), metadata={op_name="jit(foo)/jit(main)/div" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=128}
      call.41 = bf16[4,16,4,16]{3,2,1,0} call(divide.32, constant.5, constant.4), to_apply=clip.33
      convert.42 = f8e4m3fn[4,16,4,16]{3,2,1,0} convert(call.41), metadata={op_name="jit(foo)/jit(main)/convert_element_type[new_dtype=float8_e4m3fn weak_type=False]" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=130}
      convert.43 = bf16[4,16,4,16]{3,2,1,0} convert(convert.42), metadata={op_name="jit(foo)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=134}
      custom-call.4.0 = (bf16[4,4,16,16]{3,1,2,0}, u8[16]{0}) custom-call(convert.19, convert.31, convert.43), custom_call_target="__cudnn$fmhaSoftmax", operand_layout_constraints={bf16[4,16,4,16]{3,2,1,0}, bf16[4,16,4,16]{3,2,1,0}, bf16[4,16,4,16]{3,2,1,0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/dot_product_attention_fwd[scale=1.0 seed=42 dropout_rate=0.0 variadic_args=(False, False, False) mask_type=MaskType.NO_MASK layout=0 is_training=False]" source_file="/opt/rosetta/test_num.py" source_line=331}, backend_config={"operation_queue_id": "0", "wait_on_operation_queues": [], "cudnn_fmha_backend_config": {"algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"}, "is_cudnn_frontend": true, "workspace_size": "0"}, "fmha_scale": 1.0, "dropout_rate": 0.0, "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": ["4", "4", "16", "16"], "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [], "minor_to_major": ["3", "2", "1", "0"], "tiles": [], "element_size_in_bits": "0", "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID", "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"}, "is_dynamic_dimension": [false, false, false, false]}, "seed": 42, "is_flash_attention": true, "mask_type": "NO_MASK", "bmm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}}}
      get-tuple-element.5.0 = bf16[4,4,16,16]{3,1,2,0} get-tuple-element(custom-call.4.0), index=0, metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/dot_product_attention_fwd[scale=1.0 seed=42 dropout_rate=0.0 variadic_args=(False, False, False) mask_type=MaskType.NO_MASK layout=0 is_training=False]" source_file="/opt/rosetta/test_num.py" source_line=331}
      ROOT bitcast.8.0 = bf16[4,16,4,16]{3,2,1,0} bitcast(get-tuple-element.5.0), metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/dot_product_attention_fwd[scale=1.0 seed=42 dropout_rate=0.0 variadic_args=(False, False, False) mask_type=MaskType.NO_MASK layout=0 is_training=False]" source_file="/opt/rosetta/test_num.py" source_line=331}
    } // main.106
  )";
    return hlo_text;
  }

  const std::string  // NOLINT
  GetModuleFlash_Attention_Inference_BMM1_NoMask_Generation_Softmax_BMM2_HloString_FP8() {  // NOLINT
   const std::string hlo_text = R"(
    HloModule jit__unnamed_wrapped_function_, entry_computation_layout={(bf16[4,16,4,16]{3,2,1,0}, bf16[4,16,4,16]{3,2,1,0}, bf16[4,16,4,16]{3,2,1,0})->bf16[4,16,4,16]{3,2,1,0}}}, allow_spmd_sharding_propagation_to_parameters={true,true,true}, allow_spmd_sharding_propagation_to_output={true}

    clip.33 {
      Arg_2.36 = bf16[] parameter(2), metadata={op_name="jit(foo)/jit(main)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) in_layouts=(None, None, None) out_layouts=(None,) resource_env=None donated_invars=(False, False, False) name=clip keep_unused=False inline=False]"}
      broadcast.39 = bf16[4,16,4,16]{3,2,1,0} broadcast(Arg_2.36), dimensions={}, metadata={op_name="jit(foo)/jit(main)/jit(clip)/min" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=129}
      Arg_1.35 = bf16[] parameter(1), metadata={op_name="jit(foo)/jit(main)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) in_layouts=(None, None, None) out_layouts=(None,) resource_env=None donated_invars=(False, False, False) name=clip keep_unused=False inline=False]"}
      broadcast.37 = bf16[4,16,4,16]{3,2,1,0} broadcast(Arg_1.35), dimensions={}, metadata={op_name="jit(foo)/jit(main)/jit(clip)/max" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=129}
      Arg_0.34 = bf16[4,16,4,16]{3,2,1,0} parameter(0), metadata={op_name="jit(foo)/jit(main)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) in_layouts=(None, None, None) out_layouts=(None,) resource_env=None donated_invars=(False, False, False) name=clip keep_unused=False inline=False]"}
      maximum.38 = bf16[4,16,4,16]{3,2,1,0} maximum(broadcast.37, Arg_0.34), metadata={op_name="jit(foo)/jit(main)/jit(clip)/max" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=129}
      ROOT minimum.40 = bf16[4,16,4,16]{3,2,1,0} minimum(broadcast.39, maximum.38), metadata={op_name="jit(foo)/jit(main)/jit(clip)/min" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=129}
    } // clip.33


    ENTRY main.106 {
      Arg_0.1 = bf16[4,16,4,16]{3,2,1,0} parameter(0), metadata={op_name="quey"}
      constant.6 = bf16[] constant(1)
      broadcast.7 = bf16[4,16,4,16]{3,2,1,0} broadcast(constant.6), dimensions={}
      divide.8 = bf16[4,16,4,16]{3,2,1,0} divide(Arg_0.1, broadcast.7), metadata={op_name="jit(foo)/jit(main)/div" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=128}
      constant.5 = bf16[] constant(-448)
      constant.4 = bf16[] constant(448)
      call.17 = bf16[4,16,4,16]{3,2,1,0} call(divide.8, constant.5, constant.4), to_apply=clip.33
      convert.18 = f8e4m3fn[4,16,4,16]{3,2,1,0} convert(call.17), metadata={op_name="jit(foo)/jit(main)/convert_element_type[new_dtype=float8_e4m3fn weak_type=False]" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=130}
      convert.19 = bf16[4,16,4,16]{3,2,1,0} convert(convert.18), metadata={op_name="jit(foo)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=134}
      Arg_1.2 = bf16[4,16,4,16]{3,2,1,0} parameter(1), metadata={op_name="key"}
      divide.20 = bf16[4,16,4,16]{3,2,1,0} divide(Arg_1.2, broadcast.7), metadata={op_name="jit(foo)/jit(main)/div" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=128}
      call.29 = bf16[4,16,4,16]{3,2,1,0} call(divide.20, constant.5, constant.4), to_apply=clip.33
      convert.30 = f8e4m3fn[4,16,4,16]{3,2,1,0} convert(call.29), metadata={op_name="jit(foo)/jit(main)/convert_element_type[new_dtype=float8_e4m3fn weak_type=False]" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=130}
      convert.31 = bf16[4,16,4,16]{3,2,1,0} convert(convert.30), metadata={op_name="jit(foo)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=134}
      Arg_2.3 = bf16[4,16,4,16]{3,2,1,0} parameter(2), metadata={op_name="value"}
      divide.32 = bf16[4,16,4,16]{3,2,1,0} divide(Arg_2.3, broadcast.7), metadata={op_name="jit(foo)/jit(main)/div" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=128}
      call.41 = bf16[4,16,4,16]{3,2,1,0} call(divide.32, constant.5, constant.4), to_apply=clip.33
      convert.42 = f8e4m3fn[4,16,4,16]{3,2,1,0} convert(call.41), metadata={op_name="jit(foo)/jit(main)/convert_element_type[new_dtype=float8_e4m3fn weak_type=False]" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=130}
      convert.43 = bf16[4,16,4,16]{3,2,1,0} convert(convert.42), metadata={op_name="jit(foo)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/flax/flax/linen/fp8_ops.py" source_line=134}
      custom-call.4.0 = (bf16[4,4,16,16]{3,1,2,0}, u8[16]{0}) custom-call(convert.19, convert.31, convert.43), custom_call_target="__cudnn$fmhaSoftmax", operand_layout_constraints={bf16[4,16,4,16]{3,2,1,0}, bf16[4,16,4,16]{3,2,1,0}, bf16[4,16,4,16]{3,2,1,0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/dot_product_attention_fwd[scale=1.0 seed=42 dropout_rate=0.0 variadic_args=(False, False, False) mask_type=MaskType.NO_MASK layout=0 is_training=False]" source_file="/opt/rosetta/test_num.py" source_line=331}, backend_config={"operation_queue_id": "0", "wait_on_operation_queues": [], "cudnn_fmha_backend_config": {"algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"}, "is_cudnn_frontend": true, "workspace_size": "0"}, "fmha_scale": 1.0, "dropout_rate": 0.0, "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": ["4", "4", "16", "16"], "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [], "minor_to_major": ["3", "2", "1", "0"], "tiles": [], "element_size_in_bits": "0", "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID", "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"}, "is_dynamic_dimension": [false, false, false, false]}, "seed": 42, "is_flash_attention": true, "mask_type": "NO_MASK", "bmm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}}}
      get-tuple-element.5.0 = bf16[4,4,16,16]{3,1,2,0} get-tuple-element(custom-call.4.0), index=0, metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/dot_product_attention_fwd[scale=1.0 seed=42 dropout_rate=0.0 variadic_args=(False, False, False) mask_type=MaskType.NO_MASK layout=0 is_training=False]" source_file="/opt/rosetta/test_num.py" source_line=331}
      ROOT bitcast.8.0 = bf16[4,16,4,16]{3,2,1,0} bitcast(get-tuple-element.5.0), metadata={op_name="jit(<unnamed wrapped function>)/jit(main)/dot_product_attention_fwd[scale=1.0 seed=42 dropout_rate=0.0 variadic_args=(False, False, False) mask_type=MaskType.NO_MASK layout=0 is_training=False]" source_file="/opt/rosetta/test_num.py" source_line=331}
    } // main.106
  )";
    return hlo_text;
  }

  template <typename T>
  void TestImpl_Flash_Attention_Inference_BMM1_NoMask_Softmax_BMM2() {
    if (skip_reason_) GTEST_SKIP() << *skip_reason_;
    if (GetDnnVersionInfoOrDefault(backend().default_stream_executor()) <
        se::dnn::VersionInfo(8, 9, 3)) {
      GTEST_SKIP() << "Flash Attention requires cuDNN >= 8.9.3.";
    }
    XlaBuilder builder(TestName());
    // pass padding mask as bias
    std::string hlo_string =
        GetModuleFlash_Attention_Inference_BMM1_NoMask_Generation_Softmax_BMM2_HloString_BF16();  // NOLINT
    // generate padding mask in cuDNN directly
    // XLA pattern match does not support pattern matching padding mask
    // so directly lower to custom call instead for reference
    std::string hlo_string_ref =
        GetModuleFlash_Attention_Inference_BMM1_NoMask_Generation_Softmax_BMM2_HloString_FP8();  // NOLINT
    EXPECT_TRUE(RunAndCompareTwoModules(hlo_string, hlo_string_ref,
                                        ErrorSpec{1e-5, 1e-5}));
  }
};