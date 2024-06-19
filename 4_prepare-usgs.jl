using CairoMakie
using DataFrames, CSV
using Distributions
using Statistics
using LinearAlgebra, Random
using JSON
using DataInterpolations

include("./utils/makie-defaults.jl")
set_theme!(mints_theme)
update_theme!(
    figure_padding=30,
    Axis=(
        xticklabelsize=20,
        yticklabelsize=20,
        xlabelsize=22,
        ylabelsize=22,
        titlesize=25,
    ),
    Colorbar=(
        ticklabelsize=20,
        labelsize=22
    )
)


datapath = joinpath("./data/usgs/original")


spec_dict = Dict()


name_map = Dict(
    "almandine" => "Almandine",
    "ammonio-illite" => "Ammonium Illite",
    "biotite" => "Biotite",
    "carnallite" => "Carnallite",
    "ilmenite" => "Ilmenite",
    "magnetite" => "Magnetite",
    "quartz" => "Quartz",
    "rutile" => "Rutile",
    "zircon" => "Zircon"
)

@assert all(ispath.(joinpath.(datapath, keys(name_map) .* "_wl.txt")))
@assert all(ispath.(joinpath.(datapath, keys(name_map) .* "_ref.txt")))


ref_dict = Dict()

for (fname, truename) ∈ name_map
    println(fname)
    λ = parse.(Float64, readlines(joinpath(datapath, fname*"_wl.txt"))[2:end])
    ref = parse.(Float64, readlines(joinpath(datapath, fname*"_ref.txt"))[2:end])

    idx_good = findall(ref .≥ 0.0)

    # interpolate data to desired values
    λs = 0.5:0.005:2.5
    itp = LinearInterpolation(ref[idx_good], λ[idx_good])

    ref_dict[truename] = Dict(
        "λ" => λs,
        "R" => itp(λs)
    )
end



min_names = keys(ref_dict)

fig = Figure();
ax = Axis(fig[1,1], xlabel="λ (μm)", ylabel="Reflectance");
ls = []
i = 1

for min_name ∈ min_names
    lstyle = (i>7) ? :dash : :solid
    li = lines!(ax, ref_dict[min_name]["λ"], ref_dict[min_name]["R"], linewidth=2, linestyle=lstyle)

    push!(ls, li)
    i += 1
end
ylims!(ax, 0,1)
xlims!(ax, 0.5, 2.5)

fig[1,2] = Legend(fig, ls, [min_names...], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=13, height=-5)

fig




df_out = DataFrame()
df_out[!, "λ"] = collect(0.5:0.005:2.5)
df_out[!, "Carnallite"] = ref_dict["Carnallite"]["R"]
df_out[!, "Quartz"] = ref_dict["Quartz"]["R"]
df_out[!, "Biotite"] = ref_dict["Biotite"]["R"]
df_out[!, "Zircon"] = ref_dict["Zircon"]["R"]
df_out[!, "Rutile"] = ref_dict["Rutile"]["R"]
df_out[!, "Ammonium Illite"] = ref_dict["Ammonium Illite"]["R"]
df_out[!, "Almandine"] = ref_dict["Almandine"]["R"]


# visualize the data

min_names = [n for n ∈ names(df_out) if n != "λ"]
fig = Figure();
ax = Axis(fig[2,1], xlabel="λ (μm)", ylabel="Reflectance");
ls = []
i = 1

for min_name ∈ min_names
    lstyle = (i>7) ? :dash : :solid
    li = lines!(ax, df_out.λ, df_out[:, min_name], linewidth=3, linestyle=lstyle)

    push!(ls, li)
    i += 1
end
ylims!(ax, 0,1)
xlims!(ax, 0.35, 2.5)

fig[1,1] = Legend(fig, ls, [min_names...], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=14, height=-5)

save("./figures/synthetic-usgs/source-spectra.png", fig)
fig



# save the dataset
CSV.write(joinpath("./data", "usgs", "usgs.csv"), df_out)


