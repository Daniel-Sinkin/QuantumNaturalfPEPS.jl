
function _dlenv_capture_enabled(policy::Symbol)::Bool
    policy === :eager && return false
    policy === :auto || throw(ArgumentError("capture policy must be auto or eager"))
    route = get(ENV, "QNPEPS_DLENV_TRUNC", "rf")
    return route == "rf" || route == "rangefinder"
end
