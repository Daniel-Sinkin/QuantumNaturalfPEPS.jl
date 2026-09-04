module cuQuantumNaturalfPEPS

include("capi/capi.jl")
include("densitymatrix/densitymatrix.jl")
include("peps/peps.jl")
include("tensor/tensor.jl")
include("dlenv/dlenv.jl")
include("zipup/zipup.jl")
include("linalg/linalg.jl")
include("sampler/sampler.jl")
include("minsr/minsr.jl")
include("eo/eo.jl")
include("e2e/e2e.jl")
include("capi/ffi_extensions.jl")

export QnpepsConfig
export TRUNCATION_DEFAULT, TRUNCATION_DENSITY
export QnpepsElocConfig
export QnpepsElocDiagBond, QnpepsElocFlipTerm, QnpepsElocTermTable, HeisenbergTerms
export QnpepsZipupPepsRowArgs, QnpepsZipupMpoMpsDesc, QnpepsZipupMpoMpsArgs
export QnpepsDensitySettings, QnpepsDensityRankRecord, QnpepsDensityFilterArgs
export QnpepsDensityWorkspaceSizes
export QnpepsGramDesc, QnpepsGramArgs, QnpepsGramFootprint
export QnpepsMinsrDesc, QnpepsMinsrArgs
export Peps, CuPeps, CuDlenv, CuDlenvBuilt, ZipupWorkspace, DlenvArenaPlan, DlenvHost
export SamplerContext
export SamplerArenaPlan, SamplerHost, SweepPlan, SweepHost
export GramContext, MinsrContext
export MAX_BATCH_SIZE
export GRAM_CONSUMER_SLAB, GRAM_CONSUMER_CUSTOM, GRAM_CONSUMER_DENSE
export load_peps, upload_peps, random_unitary_peps
export double_layer, double_layer_step, double_layer_rowwise
export materialize_dlenv
export build_dlenv!
export plan_dlenv_host, dlenv_capture_mode, dlenv_capture_reason
export build_dlenv_device, build_dlenv_host
export zipup_mpo_mps
export ZipupDims, ZipupSettings, ZipupConfig
export DensityMatrixProductState, DensityMatrixProductOperator
export DensityMatrixTraceBuffers, DensityMatrixPlan
export densitymatrix_apply!, densitymatrix_apply, densitymatrix_rank_records
export DensityPrecision, density_precision_full
export DensityMatrixWorkspaceGeometry, DensityMatrixWorkspaceSizes
export densitymatrix_workspace_sizes
export raw_gram!, gram_footprint
export minsr_direction, minsr_direction!
export minsr_dense_count, minsr_compact_count, minsr_scratch_bytes
export sample_peps, sample_peps!, sample_peps_host, batched_rangefinder, sampler_pool_release
export plan_sampler_host, refresh_sampler!, sample_multigpu!
export sampler_sweep, sweep_capture_mode, sweep_capture_reason
export FFI
export heisenberg_terms, truncated_rydberg_terms
export EoHost, EoHostArguments, EoHostStats, EoHostBindingState, EoHostLifecycleState
export EoFlipClassification, EoTermClassification, EoTermTable
export EoSelector, EoSelectorDecision, EoSelectorAccounting
export eo_selector, eo_selector_decision, eo_selector_accounting
export eo_term_table, set_eo_selector!, advance_eo_selector!
export eo_host_selector, eo_host_term_table
export eo_host_execute!, eo_host_warm!, bind_eo_host!, seal_eo_host!
export eo_host_replace_buffers!, eo_host_stats, eo_host_binding_state
export eo_host_row_pointers, eo_host_lifecycle_state
export QnpepsE2eConfig
export vmc_step!
export VMCContext, submit_peps!, vmc_direction!, vmc_euler_step!
export E2eGramContext, e2e_raw_gram!, e2e_raw_gram, e2e_gram_footprint
export NodeStepHost, NodeStepTelemetry, node_step!, bind_node_peps!
export reset_node_step_schedule!
export MinsrHost, MinsrHostInputs, MinsrHostStats, MinsrHostError
export MinsrHostTopology, MINSR_HOST_TOPOLOGY_JURECA, MINSR_HOST_TOPOLOGY_JUPITER
export minsr_host_run!, minsr_host_try_run!
export dense_count, compact_count, eloc_compact_count
export minsr_scratch_bytes, step_scratch_bytes, step_multigpu_scratch_bytes
export node_footprint_bytes
export e2e_version, e2e_strerror, eloc_version, sampler_version
export AbstractRunAdapter, IterationResult, RunNotice
export RunEvent, RunnerConfig, RunnerControl, RunnerResult
export AbstractEventSink, StreamEventSink, JsonlEventSink, TeeEventSink, CallbackEventSink
export emit_event!, flush_event_sink!, close_event_sink!, event_record
export initialize_adapter!, step_adapter!, checkpoint_adapter!, close_adapter!
export adapter_iteration, adapter_status
export request_command!, parse_runner_command, start_stdin_control!, run_iterations!
export SyntheticAdapter

end
