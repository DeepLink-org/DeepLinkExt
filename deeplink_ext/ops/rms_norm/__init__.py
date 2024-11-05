_not_impl = "[deeplink_ext] {op_name} is not implemented in diopi. Falling back to the slower torch implementation."

try:
    from .easyllm_rms_norm import rms_norm
except:
    print(
        _not_impl.format(op_name="RMSNorm"),
    )
    from .easyllm_rms_norm_fallback import rms_norm_torch as rms_norm

try:
    from .internevo_rms_norm import MixedFusedRMSNorm
except:
    print(
        _not_impl.format(op_name="RMSNorm"),
    )
    from .internevo_rms_norm_fallback import MixedRMSNormTorch as MixedFusedRMSNorm
