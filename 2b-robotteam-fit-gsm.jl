using MLJ, GenerativeTopographicMapping
using DataFrames, CSV
using StableRNGs
using Random
using JSON

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


outpath = "./output/robot-team"
figpath = "./figures/robot-team"
mkpath(outpath)
mkpath(figpath)


include("./utils/fit-metrics.jl")


# LOAD IN  COMBINED DATASET
readdir("./data/robot-team/dye-test")
datapath = "./data/robot-team/df_features.csv"
df = CSV.read(datapath, DataFrame);


# Explore data and infer intrinsic dimensionality (i.e. # of endmembers)
using LinearAlgebra

size(Array(df))

Svd = svd(Array(df) .- mean(Array(df), dims=1));

σs = abs2.(Svd.S) ./ (nrow(df) - 1)

variances =  σs ./ sum(σs)
cum_var = cumsum(σs ./ sum(σs))

fig = Figure();
ax = Axis(fig[1,1], xlabel="component", ylabel="Explained Variance", xticks=(1:length(σs)));
lines!(ax, 1:length(σs), variances, linewidth=2, linestyle=:dash)
scatter!(ax, 1:length(σs), variances, markersize=25)
xlims!(ax, 0.85, 10)
ylims!(ax, -0.01, nothing)
fig

save(joinpath(figpath, "pca-variance.png"), fig)


# Based on this, between 4-8 endmembers should be fine
# We can try to justify this 2 different ways:
#  1. BIC/AIC
#  2. Distribution(s) of abundances.

GSMMultUpLinear()
GSMMultUpNonlinear()







