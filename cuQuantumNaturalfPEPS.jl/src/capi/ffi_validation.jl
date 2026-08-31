import .._capi_version_mismatch_error
import .._compiled_capi_version
import .._component_version_mismatch_error
import ..EXPECTED_CAPI_VERSION
import ..EXPECTED_E2E_VERSION
import ..EXPECTED_ELOC_VERSION
import .._canonical_lib_path

function validate_library()::Nothing
    path = _canonical_lib_path()
    got = unsafe_string(ccall((:qnpeps_capi_version, _canonical_lib_path()), Cstring, ()))
    got_version = _compiled_capi_version(got)
    got_version == EXPECTED_CAPI_VERSION || _capi_version_mismatch_error(; path, got, got_version)
    e2e = unsafe_string(ccall((:qnpeps_e2e_version, _canonical_lib_path()), Cstring, ()))
    e2e == EXPECTED_E2E_VERSION || _component_version_mismatch_error(;
        path,
        component="e2e",
        got=e2e,
        expected=EXPECTED_E2E_VERSION,
    )
    eloc = unsafe_string(ccall((:qnpeps_eloc_version, _canonical_lib_path()), Cstring, ()))
    eloc == EXPECTED_ELOC_VERSION || _component_version_mismatch_error(;
        path,
        component="eo",
        got=eloc,
        expected=EXPECTED_ELOC_VERSION,
    )
    return nothing
end
