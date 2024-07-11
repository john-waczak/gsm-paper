using DataFrames, CSV
using Distributions
using Random, Statistics, LinearAlgebra
using StableRNGs


using CairoMakie, TernaryDiagrams

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



rng = StableRNG(42)


# set up paths

datapath = "./data/1_usgs"
if !ispath(datapath)
    mkpath(datapath)
end

if !ispath(joinpath(datapath, "linear"))
    mkpath(joinpath(datapath, "linear"))
end

if !ispath(joinpath(datapath, "bilinear"))
    mkpath(joinpath(datapath, "bilinear"))
end


figpath = "./figures/1_usgs"
if !ispath(figpath)
    mkpath(figpath)
end

if !ispath(joinpath(figpath, "linear"))
    mkpath(joinpath(figpath, "linear"))
end

if !ispath(joinpath(figpath, "bilinear"))
    mkpath(joinpath(figpath, "bilinear"))
end


outpath = "./output/1_usgs"
if !ispath(outpath)
    mkpath(outpath)
end

if !ispath(joinpath(outpath, "linear"))
    mkpath(joinpath(outpath, "linear"))
end

if !ispath(joinpath(outpath, "bilinear"))
    mkpath(joinpath(outpath, "bilinear"))
end




# ---------------------------------------------------
#
#
#
# we have 2 tests to do for the USG data
# 1. Linear Mixing
# 2. Linear Mixing + Bilinear Mixing
#
#
#
# ---------------------------------------------------


SNR = [0, 5, 10, 15, 20, 25, 30, 35, Inf]


# load in source spectra
df = CSV.read("./data/src/usgs/usgs.csv", DataFrame)

λs = df.λ
# use mineral types from original VCA paper
# 10.1109/TGRS.2005.844293
min_to_use = ["Carnallite", "Ammonium Illite", "Biotite"]

# add 1.0 to each spectrum so nothing is negative
R1 = df[:, min_to_use[1]]
R2 = df[:, min_to_use[2]]
R3 = df[:, min_to_use[3]]


# set up number of points and parameters for
# sampling to also match the VCA paper
Npoints = 1_000
α_true = [1/3, 1/3, 1/3]
f_dir = Dirichlet(α_true)
abund = rand(rng, f_dir, Npoints)

# generate linear combinations
Xbase = zeros(Npoints, length(λs));
for i ∈ axes(Xbase, 1)
    Xbase[i,:] .= (abund[1,i] .* R1) .+ (abund[2,i] .* R2) .+ (abund[3,i] .* R3)
end


colnames = Symbol.(["λ_" * lpad(i, 3, "0") for i ∈ 1:length(λs)])
for snr ∈ SNR
    X = copy(Xbase)

    σnoise = isinf(snr) ? 0.0 : sqrt(mean(X.^2)/(10^(snr/10)))

    println("SNR = $(snr)\t σ = $(σnoise)")
    if !isinf(snr)
        # add in the noise
        for i ∈ axes(X, 1)
            X[i,:] .= X[i,:] .+ rand(rng, Normal(0, σnoise), length(λs))
        end
    end

    X .= max.(X, 0.0)
    @assert all(X .≥ 0.0)

    # save linear dataset
    dfX = DataFrame(X, colnames);
    CSV.write(joinpath(datapath, "linear", "df_snr-$(string(snr))_std-$(round(σnoise, digits=4)).csv"), dfX)

    # add in bilinear terms
    for i ∈ axes(X, 1)
        # add in bilinear mixing
        X[i,:] .= X[i,:] .+ (abund[1,i] * abund[2,i] .* R1 .* R2) + (abund[1,i] * abund[3,i] .* R1 .* R3) + (abund[2,i] * abund[3,i] .* R2 .* R3)
    end

    # save bilinear dataset
    dfX = DataFrame(X, colnames);
    CSV.write(joinpath(datapath, "bilinear", "df_snr-$(string(snr))_std-$(round(σnoise, digits=4)).csv"), dfX)
end




# save the abundances to a dataframe
size(abund)
df_abund = DataFrame(
    Dict(
        "R1" => abund[1,:],
        "R2" => abund[2,:],
        "R3" => abund[3,:],
    )
)

CSV.write(joinpath(datapath, "df_abund.csv"), df_abund)


# Plot abundance distributions
fig = Figure();
ax = Axis(fig[1, 1], aspect=1);

ternaryaxis!(
    ax,
    hide_triangle_labels=false,
    hide_vertex_labels=false,
    labelx_arrow = "Red",
    label_fontsize=20,
    tick_fontsize=15,
)

ternaryscatter!(
    ax,
    abund[1,:],
    abund[2,:],
    abund[3,:],
    color=[CairoMakie.RGBf(abund[:,i]...) for i ∈ 1:Npoints],
    marker=:circle,
    markersize = 8,
)

# the triangle is drawn from (0,0) to (0.5, sqrt(3)/2) to (1,0).
xlims!(ax, -0.2, 1.2) # to center the triangle and allow space for the labels
ylims!(ax, -0.3, 1.1)
hidedecorations!(ax) # to hide the axis decorations

text!(ax, Point2f(-0.01, 0.5), text=min_to_use[3], fontsize=22)
text!(ax, Point2f(0.83, 0.45), text=replace(min_to_use[2], " " => "\n"), fontsize=22)
text!(ax, Point2f(0.35, -0.175), text=min_to_use[1], fontsize=22)

save(joinpath(figpath, "abundance-orig.png"), fig)

fig

# create plot of original endmembers
fig = Figure();
ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");

l1 = lines!(ax, λs, R1, color=mints_colors[2], linewidth=3)
l2 = lines!(ax, λs, R2, color=mints_colors[1], linewidth=3)
l3 = lines!(ax, λs, R3, color=mints_colors[3], linewidth=3)

fig[1,1] = Legend(fig, [l1, l2, l3], [min_to_use...,], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=22, height=-5)
xlims!(ax, λs[1], λs[end])

save(joinpath(figpath, "endmembers-orig.png"), fig)

fig


