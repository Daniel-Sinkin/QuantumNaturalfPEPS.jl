#!/usr/bin/env julia

using cuQuantumNaturalfPEPS

include(joinpath(@__DIR__, "common.jl"))

function usage(io::IO = stdout)
    println(io, "Synthetic headless runner demo")
    println(io, "  --iterations N")
    println(io, "  --jsonl PATH")
    println(io, "  --checkpoint PATH")
    println(io, "  --resume")
    println(io, "  --interactive")
    println(io, "  --checkpoint-every N")
    println(io, "  --sleep-ms N")
    println(io, "  --warn-at N")
    println(io, "  --fail-at N")
    println(io, "  --debug")
    return nothing
end

function main(arguments)
    options = parse_app_options("runner_demo.jl", arguments)
    app_dry_run("runner_demo.jl", options) && return 0
    if options[:help]
        usage()
        return 0
    end
    adapter = SyntheticAdapter(
        checkpoint_path = options[:checkpoint],
        resume = options[:resume],
        warning_iterations = Set(options[:warn_at]),
        failure_iteration = options[:fail_at],
        sleep_seconds = options[:sleep_ms] / 1000,
    )
    terminal = StreamEventSink(stdout)
    sink = terminal
    jsonl = nothing
    if options[:jsonl] !== nothing
        jsonl = JsonlEventSink(options[:jsonl])
        sink = TeeEventSink(terminal, jsonl)
    end
    control = RunnerControl()
    options[:interactive] && start_stdin_control!(control)

    result = try
        run_iterations!(
            adapter,
            RunnerConfig(
                iterations = options[:iterations],
                checkpoint_every = options[:checkpoint_every],
                expensive_debug = options[:debug],
            );
            sink = sink,
            control = control,
        )
    finally
        jsonl === nothing || close_event_sink!(jsonl)
    end
    println(
        stdout,
        "[runner] status=",
        result.status,
        " completed=",
        result.iterations_completed,
        " last_iteration=",
        result.last_iteration,
    )
    return result.status == :error ? 1 : 0
end

abspath(PROGRAM_FILE) == (@__FILE__) && exit(main(ARGS))
