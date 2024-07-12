println("Starting GSM fit script")
println("\tLoading packages...")
using MLJ, GenerativeTopographicMapping
using DataFrames, CSV
using StableRNGs
using Random
using JSON

include("./utils/robot-team-config.jl")

using CairoMakie
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



println("\tLoading data...")
outpath = "./output/robot-team/dye"
figpath = "./figures/robot-team/dye"
mkpath(outpath)
mkpath(figpath)

include("./utils/fit-metrics.jl")

# LOAD IN  COMBINED DATASET
readdir("./data/robot-team/dye-test")
readdir("./data/robot-team/")

df_features = CSV.read("./data/robot-team/dye-test/df_features-nw.csv", DataFrame);
# df_features = CSV.read("./data/robot-team/dye-test/df_features-nw.csv", DataFrame);
# df_targets = CSV.read("./data/robot-team/dye-test/df_targets-nw.csv", DataFrame);



# Explore data and infer intrinsic dimensionality (i.e. # of endmembers)
println("\tAssessing dimensionality...")

using LinearAlgebra
Svd = svd(Array(df_features) .- mean(Array(df_features), dims=1));
σs = abs2.(Svd.S) ./ (nrow(df_features) - 1)
variances =  σs ./ sum(σs)
cum_var = cumsum(σs ./ sum(σs))

fig = Figure();
ax = Axis(fig[1,1], xlabel="component", ylabel="Explained Variance", xticks=(1:length(σs)), xminorgridvisible=false);
lines!(ax, 1:length(σs), variances, linewidth=2, linestyle=:dash)
scatter!(ax, 1:length(σs), variances, markersize=25)
xlims!(ax, 0.85, 10)
ylims!(ax, -0.01, nothing)
fig

save(joinpath(figpath, "pca-variance.png"), fig)



# Based on this, between 4-8 endmembers should be fine

rand(nrow(df_features), n_nodes)

n_nodes = 3000
n_rbfs = 500

Nv_s = [3, 4, 5, 6] #, 7, 8]
s_s = [0.05, 0.1, 0.25]
λe_s = [0.01, 0.001, 0.0001]
λw_s = [100, 10, 1, 0.1]

println("\tNumber of models to train: ", length(Nv_s) * length(s_s) * length(λe_s) * length(λw_s))
println("\tTraiing models...")

idx = 1
for Nv ∈ Nv_s
    for s ∈ s_s
        for λe ∈ λe_s
            for λw ∈ λw_s
                println("\n")
                println("Nv: ", Nv, "\ts: ", s, "\tλe: ", λe, "\tλw: ", λw)
                println("\n")

                gsm = GSMBigCombo(n_nodes=n_nodes, n_rbfs=n_rbfs, s=s, Nv=Nv, niters=50, tol=1e-6, nepochs=100)

                mach = machine(gsm, df_features)
                fit!(mach, verbosity=1)

                # generate report
                rpt = report(mach)
                rpt["Nv"] = Nv
                rpt["s"] = s
                rpt["λe"] = λe
                rpt["λw"] = λw

                open(joinpath(outpath, "fit_" * lpad(idx, 3, "0")) *".json", "w") do f
                    JSON.print(f, rpt)
                end

                idx += 1
            end
        end
    end
end

