using CSV
using DataFrames
using Plots
using Statistics
using JSON



database = Dict(
    "Baseline" => [
        "ablation-tracking_error-20230808T15-27-50.csv",
        "ablation-tracking_error-20230808T15-29-45.csv",
        "ablation-tracking_error-20230808T15-33-19.csv",
        "ablation-tracking_error-20230808T15-43-29.csv",
        "ablation-tracking_error-20230808T21-14-38.csv",
        "ablation-tracking_error-20230808T21-18-24.csv",
        "ablation-tracking_error-20230808T21-12-06.csv",
        "ablation-tracking_error-20230808T21-00-10.csv",
        "ablation-tracking_error-20230808T21-03-38.csv",
        "ablation-tracking_error-20230808T15-56-32.csv",
    ],
    "Curriculum" => [
        "ablation-tracking_error-20230809T18-25-14.csv",
        "ablation-tracking_error-20230809T18-29-43.csv",
        "ablation-tracking_error-20230809T18-39-30.csv",
        "ablation-tracking_error-20230809T18-41-30.csv",
        "ablation-tracking_error-20230809T18-46-06.csv",
        "ablation-tracking_error-20230809T18-48-57.csv",
        "ablation-tracking_error-20230809T19-01-35.csv",
        "ablation-tracking_error-20230809T18-51-38.csv",
        "ablation-tracking_error-20230809T18-57-50.csv",
        "ablation-tracking_error-20230809T18-58-33.csv",
    ],
    "Action History" => [
        "ablation-tracking_error-20230808T16-38-13.csv",
        "ablation-tracking_error-20230808T16-42-40.csv",
        "ablation-tracking_error-20230808T16-45-04.csv",
        "ablation-tracking_error-20230808T16-47-24.csv",
        "ablation-tracking_error-20230808T16-49-31.csv",
        "ablation-tracking_error-20230808T16-51-11.csv",
        "ablation-tracking_error-20230808T16-52-47.csv",
        "ablation-tracking_error-20230808T16-55-11.csv",
        "ablation-tracking_error-20230808T16-56-36.csv",
        "ablation-tracking_error-20230808T16-59-29.csv",
    ],
    "Asymmetric Actor-Critic" => [
        "ablation-tracking_error-20230808T19-17-18.csv",
        "ablation-tracking_error-20230808T19-19-51.csv",
        "ablation-tracking_error-20230808T19-22-31.csv",
        "ablation-tracking_error-20230808T19-25-15.csv",
        "ablation-tracking_error-20230808T19-27-03.csv",
        "ablation-tracking_error-20230808T19-29-37.csv",
        "ablation-tracking_error-20230808T19-32-13.csv",
        "ablation-tracking_error-20230808T19-35-19.csv",
        "ablation-tracking_error-20230808T19-37-51.csv",
        "ablation-tracking_error-20230808T19-40-51.csv",
    ],
    "Observation Noise" => [
        "ablation-tracking_error-20230808T19-44-11.csv",
        "ablation-tracking_error-20230808T19-48-54.csv",
        "ablation-tracking_error-20230808T19-51-52.csv",
        "ablation-tracking_error-20230808T19-54-35.csv",
        "ablation-tracking_error-20230808T19-57-54.csv",
        "ablation-tracking_error-20230808T20-02-10.csv",
        "ablation-tracking_error-20230808T20-05-29.csv",
        "ablation-tracking_error-20230808T20-08-38.csv",
        "ablation-tracking_error-20230808T20-11-59.csv",
        "ablation-tracking_error-20230808T20-16-44.csv",
    ],
    "Disturbance" => [
        "ablation-tracking_error-20230808T21-35-41.csv",
        "ablation-tracking_error-20230808T20-24-16.csv",
        "ablation-tracking_error-20230808T20-28-20.csv",
        "ablation-tracking_error-20230808T20-37-13.csv",
        "ablation-tracking_error-20230808T20-34-16.csv",
        "ablation-tracking_error-20230808T20-39-57.csv",
        "ablation-tracking_error-20230808T20-42-43.csv",
        "ablation-tracking_error-20230808T20-45-24.csv",
        "ablation-tracking_error-20230808T20-48-31.csv",
        "ablation-tracking_error-20230808T20-50-57.csv",
    ],
    "Curriculum Target" => [
        "ablation-tracking_error-20230809T22-34-30.csv",
        "ablation-tracking_error-20230809T22-35-21.csv",
        "ablation-tracking_error-20230809T22-44-51.csv",
        "ablation-tracking_error-20230809T22-47-45.csv",
        "ablation-tracking_error-20230809T22-51-02.csv",
        "ablation-tracking_error-20230809T22-55-44.csv",
        "ablation-tracking_error-20230809T22-58-03.csv",
        "ablation-tracking_error-20230809T23-00-28.csv",
        "ablation-tracking_error-20230809T23-02-57.csv",
        "ablation-tracking_error-20230809T23-14-14.csv",
    ]
)



function analyze(flights_names; truncate_flight_times=40)

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
        println("$(d[:name]) length: $(end_index - start_index) samples")
        flying_time = (df."Timestamp"[end_index] - df."Timestamp"[start_index]) ./ 1000.0
        flying_time = clamp(flying_time, 0, truncate_flight_times)

        # plt = plot(cs)
        plt = plot([start_index, end_index], [3, 3])
        title!(plt, d[:name])
        plot!(plt, sm)
        display(plt)

        df_trunc = df[start_index:end_index, :]
        d_out = deepcopy(d)
        d_out[:df] = df_trunc
        d_out[:time] = flying_time
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
        mse = sqrt(mean(df."bptte.x".^2 .+ df."bptte.y".^2))
        if d[:time] < truncate_flight_times
            return mse
        else
            return mse
        end
    end
    Dict(
        :rmses => RMSEs, #filter(x->!isnothing(x), RMSEs),
        :flight_times => [d[:time] for d in flight_data_processed]
    )
end





# flights_names = database["Baseline"]
# flights_names = database["Action History"]
# flights_names = database["Rotor Delay"]
# flights_names = database["Observation Noise"]
# flights_names = database["Disturbance"]

analyzed_data = Dict([(k, analyze(v)) for (k, v) in database])

mse_medians = Dict([(k, median(v[:rmses])) for (k, v) in analyzed_data])
mse_stds = Dict([(k, std(v[:rmses])) for (k, v) in analyzed_data])
mse_means = Dict([(k, mean(v[:rmses])) for (k, v) in analyzed_data])
mse_mins = Dict([(k, minimum(v[:rmses])) for (k, v) in analyzed_data])

flight_times_medians = Dict([(k, median(v[:flight_times])) for (k, v) in analyzed_data])
flight_times_means = Dict([(k, mean(v[:flight_times])) for (k, v) in analyzed_data])
# flight_times_table = Dict([(k, sort(v[:flight_times])) for (k, v) in analyzed_data])

results = JSON.json(Dict(
    "rmses" => rmses,
    "medians" => medians,
    "stds" => stds,
    "means" => means
))

open("ablation_results/results.json", "w") do f
    write(f, results)
end