using CSV
using DataFrames
using Plots
using Statistics



flights_names = [
    "ablation-tracking_error-20230808T15-27-50.csv",
    "ablation-tracking_error-20230808T15-29-45.csv",
    "ablation-tracking_error-20230808T15-33-19.csv",
    "ablation-tracking_error-20230808T15-43-29.csv",
    # "ablation-tracking_error-20230808T15-46-15.csv",
    # "ablation-tracking_error-20230808T15-49-23.csv",
    nothing,
    nothing,
    nothing,
    "ablation-tracking_error-20230808T15-51-27.csv",
    "ablation-tracking_error-20230808T15-53-57.csv",
    "ablation-tracking_error-20230808T15-56-32.csv",
]


flight_data_raw = map(flights_names) do name
    if isnothing(name)
        return nothing
    else
        Dict(:name => name, :df => CSV.read("ablation_results/logs/$name", DataFrame))
    end
end




flight_data_processed = map(flight_data_raw) do d
    if isnothing(d)
        return nothing
    end
    df = d[:df]
    # df = flight_data_raw[1]
    if isnothing(df)
        return nothing
    end
    sm = df."bptrp.sm"
    rising_edge = [false, (sm[2:end] .- sm[1:end-1] .> 0)...]
    cs = cumsum(rising_edge)
    sequences = filter(x->x != 0, unique(cs))
    sequence_lengths = map(sequences) do s
        sum(cs .== s)
    end
    start_index = findfirst(cs .== sequences[argmax(sequence_lengths)])
    end_index = start_index + argmax(sm[start_index:end] .== 0) - 2

    # plt = plot(cs)
    plt = plot([start_index, end_index], [3, 3])
    title!(plt, d[:name])
    plot!(plt, sm)
    display(plt)

    df_trunc = df[start_index:end_index, :]
    d_out = deepcopy(d)
    d_out[:df] = df_trunc
    d_out
end


RMSEs = map(flight_data_processed) do d
# df = flight_data_processed[1]
    if isnothing(d)
        return nothing
    end
    df = d[:df]
    if(isnothing(df))
        return nothing
    end
    plt = plot(df."bptte.x", df."bptte.y")
    title!(plt, d[:name])
    display(plt)
    mse = mean(df."bptte.x".^2 .+ df."bptte.y".^2)
end




