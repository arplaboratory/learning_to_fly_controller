using DataFrames
using CSV
using ProgressMeter

begin
    files = []

    for logdir in readdir("/home/jonas/.config/cfclient/logdata", join=true)
        push!(files, readdir(logdir, join=true)...)
    end

    files

    files = sort(filter(x->x[end-3:end] == ".csv", files))

    logs = @showprogress map(files) do f
        f, CSV.read(f, DataFrame)
    end

    logs = filter(x -> hasproperty(x[2], "bptrp.sm"), logs)

    for (file, df) in logs
        println("$file: $(sum(df."bptrp.sm"))")
    end
    println("Will copy $(logs[end][1]) to ablation_results/logs")
end

begin
last_n = 1
cp(logs[end - last_n + 1][1], "ablation_results/logs/$(basename(logs[end - last_n + 1][1]))")
end


