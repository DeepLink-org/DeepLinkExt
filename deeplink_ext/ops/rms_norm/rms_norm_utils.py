_not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."


def patch_RMSNorm(MixedFusedRMSNorm):
    try:
        import apex.normalization.fused_layer_norm as fused_layer_norm
    except Exception as e:
        print("Unable to import fused_layer_norm, skip mocking fused_layer_norm")
        return

    fused_layer_norm.MixedFusedRMSNorm = MixedFusedRMSNorm


def import_RMSNorm():
    try:
        from .internevo_rms_norm import MixedFusedRMSNorm
    except:
        print(
            _not_impl.format(op_name="RMSNorm"),
        )
        from .internevo_rms_norm_fallback import MixedRMSNormTorch as MixedFusedRMSNorm
    return MixedFusedRMSNorm
