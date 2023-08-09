files = readdir("ablation_data_9", join=true)

parsed = map(files) do file
    m = match(r"multirotor_td3_([^_]+)_([0-9]+)$", file)
    if isnothing(m)
        return nothing
    else
        return (file, m[1], m[2])
    end
end

parsed = filter(!isnothing, parsed)

for file in parsed
    (path, env, seed) = file

    for checkpoint in readdir(path, join=true)
        m = match(r"actor_([0-9]+).h$", checkpoint)
        println(m)
        @assert !isnothing(m)
        new_path = joinpath("ablation_data_organized_9", env, m[1])
        println("making path $new_path")
        mkpath(new_path)

        println("copying $path to $new_path")
        cp(checkpoint, joinpath(new_path, "$(seed).h"))
    end
end