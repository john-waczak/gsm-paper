using CairoMakie
using MLJ, GenerativeTopographicMapping
using DataFrames, CSV
using Distributions
using Statistics
using LinearAlgebra, Random
using StableRNGs

include("./utils/hysu.jl")

rng = StableRNG(42)

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


datapath = "./data/hysu/all.csv"
@assert ispath(datapath)

figpath = "./figures/hysu"
if !isdir(figpath)
    mkdir(figpath)
end


df = CSV.read(datapath, DataFrame)
X = Array(df)

# 1. Visualize the dataset
fig = Figure();
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance")
let
    for i ∈ axes(X, 1)
        lines!(ax, λs, X[i,:], color=(mints_colors[3], 0.5), linewidth=1)
    end

    lines!(ax, λs, mean(X, dims=1)[:], color=mints_colors[2], linewidth=3)
end
xlims!(ax, λs[1], λs[end])
ylims!(ax, 0, 1)
fig
save(joinpath(figpath, "spectra-all.png"), fig)
save(joinpath(figpath, "spectra-all.pdf"), fig)


# 2. Now let's examine the intrinsic dimensionality of the dataset
size(X)
Svd = svd(X' .- mean(X, dims=1)[:])
Svd.S

pca_vars = (abs2.(Svd.S) ./ (nrow(df)-1))

fig = Figure();
ax = Axis(fig[1,1], xlabel="Singular Value", ylabel="Percent Explained Variance");
ax2 = fig[1,1] = Axis(fig, ylabel = "Total Explained Variance");
linkxaxes!(ax, ax2)
linkyaxes!(ax, ax2)

ax2.yaxisposition = :right
ax2.yticklabelalign = (:left, :center)
ax2.xticklabelsvisible = false
ax2.xticklabelsvisible = false
ax2.xlabelvisible = false
ax2.yticklabelsvisible = false

lines!(ax, 1:length(pca_vars), pca_vars./sum(pca_vars), linewidth=3, color=mints_colors[3])
lines!(ax2, 1:length(pca_vars), cumsum(pca_vars)./sum(pca_vars), linewidth=3, color=mints_colors[1])
xlims!(ax2, 1, 10)
fig

save(joinpath(figpath, "pca-explained-variance.png"), fig)
save(joinpath(figpath, "pca-explained-variance.pdf"), fig)



# Linear Unmixing on "All" Dataset
k = 10
m = 5
λ = 0.0001
Nᵥ = 10
binomial(k + Nᵥ - 2, Nᵥ - 1)

gsm = GSM(k=k, m=m, Nv=Nᵥ, λ=λ, nonlinear=false, linear=true, bias=false, make_positive=true, tol=1e-5, nepochs=500, rand_init=false, rng=StableRNG(42))

mach = machine(gsm, df)
fit!(mach, verbosity=1)

abund = DataFrame(MLJ.transform(mach, df));


# visualize abundances so we can see if there are any useless dimensions
fig = Figure();
ax = Axis(fig[1,1], xlabel="Endmember", ylabel="log10 Abundance", xticks=(1:10, ["z₁", "z₂", "z₃", "z₄", "z₅", "z₆", "z₇", "z₈", "z₉", "z₁₀"]));
for i ∈ 1:size(abund, 2)
    cat = i*ones(Int, size(abund, 1))
    bp = boxplot!(ax, cat, log10.(abund[:,i]))
    # bp = violin!(ax, cat, abund[:,i], show_median=true)
end
fig

save(joinpath(figpath, "extra-endmembers-boxplot.png"), fig)
save(joinpath(figpath, "extra-endmembers-boxplot.pdf"), fig)



rpt = report(mach)
node_means = rpt[:node_data_means]
Q = rpt[:Q]
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]
stdev = sqrt(rpt[:β⁻¹])


# plot log likelihoods
fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Log-likelihood", yscale=log10)
lines!(ax, 2:length(llhs), llhs[2:end], linewidth=3)
fig

save(joinpath(figpath, "llh.png"), fig)
save(joinpath(figpath, "llh.pdf"), fig)


fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Q (a.u.)", yscale=log10)
lines!(ax, 2:length(Q), Q[2:end], linewidth=3)
fig

save(joinpath(figpath, "Q-fit.png"), fig)
save(joinpath(figpath, "Q-fit.pdf"), fig)


# plot endmembers
fig = Figure();
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance");

Ncolors = length(mints_colors)

i = 1
for idx ∈ idx_vertices
    Rout = node_means[:,idx]
    band!(ax, λs, Rout .- (2*stdev), Rout .+ (2*stdev), color=(mints_colors[i%Ncolors+1], 0.5), transparency=true)
    li = lines!(ax, λs, Rout, color=mints_colors[i%Ncolors+1], linewidth=2)
    i += 1
end
xlims!(ax, λs[1], λs[end])
ylims!(ax, 0, 0.9)
fig


save(joinpath(figpath, "extracted-endmembers.png"), fig)
save(joinpath(figpath, "extracted-endmembers.pdf"), fig)


# maybe a radial plot with abundances on on spokes?
