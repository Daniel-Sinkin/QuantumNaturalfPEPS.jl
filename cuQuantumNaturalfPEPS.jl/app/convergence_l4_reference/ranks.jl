function telemetry_rows(path::AbstractString, offset::Integer, step::Int)
    isfile(path) || return NamedTuple[], Int64(offset)
    lines = String[]
    next_offset = open(path, "r") do io
        seek(io, offset)
        append!(lines, readlines(io))
        return position(io)
    end
    selected = Dict{Tuple{String,String,Int},NamedTuple}()
    priority = Dict("route_cap" => 0, "rdiag" => 1, "condition" => 2, "effective_rank" => 3)
    for line in lines
        startswith(line, "call,") && continue
        fields = split(line, ','; keepempty=true)
        length(fields) == 18 || continue
        kind = fields[6]
        source =
            kind == "effective_rank" ? "effective_rank" :
            kind == "condition" ? "condition" :
            kind == "rdiag" ? "rdiag" :
            startswith(kind, "rf_route_") ? "route_cap" : ""
        isempty(source) && continue
        lane = source == "route_cap" ? -1 : parse(Int, fields[7])
        retained = source == "route_cap" ? parse(Int, fields[9]) : parse(Int, fields[8])
        row = (
            step,
            phase=fields[3],
            cut=fields[5],
            lane,
            retained,
            cap=parse(Int, fields[9]),
            batch=parse(Int, fields[10]),
            route=fields[4],
            source,
        )
        key = (row.phase, row.cut, row.lane)
        previous = get(selected, key, nothing)
        if previous === nothing || priority[row.source] > priority[previous.source]
            selected[key] = row
        end
    end
    return sort!(collect(values(selected)); by=row -> (row.phase, row.cut, row.lane)), next_offset
end

function write_rank_rows(io, rows)
    for row in rows
        write_csv_row(
            io,
            (
                row.step,
                row.phase,
                row.cut,
                row.lane,
                row.retained,
                row.cap,
                row.batch,
                row.route,
                row.source,
            ),
        )
    end
    return nothing
end
