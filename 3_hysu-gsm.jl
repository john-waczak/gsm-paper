using CairoMakie
using MLJ, GenerativeTopographicMapping
using DataFrames, CSV
using Distributions
using Statistics
using LinearAlgebra, Random
using StableRNGs
using JSON

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


datapath_all = "./data/hysu/all.csv"
datapath_large = "./data/hysu/large.csv"
datapath_small = "./data/hysu/small.csv"
@assert ispath(datapath_all)
@assert ispath(datapath_large)
@assert ispath(datapath_small)

figpath = "./figures/hysu"
if !isdir(figpath)
    mkdir(figpath)
    mkdir(joinpath(figpath, "all"))
    mkdir(joinpath(figpath, "large"))
    mkdir(joinpath(figpath, "small"))
end



df_ref = CSV.read("./data/hysu/reference-spectra.csv", DataFrame)
df_svc = CSV.read("./data/hysu/svc-spectra.csv", DataFrame)



df_all = CSV.read(datapath_all, DataFrame)
df_large = CSV.read(datapath_large, DataFrame)
df_small = CSV.read(datapath_small, DataFrame)

X_all = Array(df_all)
X_large = Array(df_large)
X_small = Array(df_small)

# 1. Visualize the dataset
fig = Figure();
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance")
let
    for i ∈ axes(X_all, 1)
        lines!(ax, λs, X_all[i,:], color=(mints_colors[3], 0.5), linewidth=1)
    end

    lines!(ax, λs, mean(X_all, dims=1)[:], color=mints_colors[2], linewidth=3)
end
xlims!(ax, λs[1], λs[end])
ylims!(ax, 0, 1)
fig
save(joinpath(figpath, "all", "spectra-all.png"), fig)
save(joinpath(figpath, "all", "spectra-all.pdf"), fig)


fig = Figure();
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance")
let
    for i ∈ axes(X_large, 1)
        lines!(ax, λs, X_large[i,:], color=(mints_colors[3], 0.5), linewidth=1)
    end

    lines!(ax, λs, mean(X_large, dims=1)[:], color=mints_colors[2], linewidth=3)
end
xlims!(ax, λs[1], λs[end])
ylims!(ax, 0, 1)
fig
save(joinpath(figpath, "large", "spectra-all.png"), fig)
save(joinpath(figpath, "large", "spectra-all.pdf"), fig)



fig = Figure();
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance")
let
    for i ∈ axes(X_small, 1)
        lines!(ax, λs, X_small[i,:], color=(mints_colors[3], 0.5), linewidth=1)
    end

    lines!(ax, λs, mean(X_small, dims=1)[:], color=mints_colors[2], linewidth=3)
end
xlims!(ax, λs[1], λs[end])
ylims!(ax, 0, 1)
fig
save(joinpath(figpath, "small", "spectra-all.png"), fig)
save(joinpath(figpath, "small", "spectra-all.pdf"), fig)





# 2. Now let's examine the intrinsic dimensionality of the dataset
size(X_all)
Svd = svd(X_all' .- mean(X_all, dims=1)[:])
Svd.S
pca_vars_all = (abs2.(Svd.S) ./ (nrow(df_all)-1))

size(X_large)
Svd = svd(X_large' .- mean(X_large, dims=1)[:])
Svd.S
pca_vars_large= (abs2.(Svd.S) ./ (nrow(df_large)-1))

size(X_small)
Svd = svd(X_small' .- mean(X_small, dims=1)[:])
Svd.S
pca_vars_small= (abs2.(Svd.S) ./ (nrow(df_small)-1))



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

lines!(ax, 1:length(pca_vars_all), pca_vars_all./sum(pca_vars_all), linewidth=2, color=mints_colors[3], linestyle=:solid)
lines!(ax, 1:length(pca_vars_large), pca_vars_large./sum(pca_vars_large), linewidth=3.5, color=mints_colors[3], linestyle=:dot)
lines!(ax, 1:length(pca_vars_small), pca_vars_small./sum(pca_vars_small), linewidth=2, color=mints_colors[3], linestyle=:dash)

lines!(ax2, 1:length(pca_vars_all), cumsum(pca_vars_all)./sum(pca_vars_all), linewidth=2, color=mints_colors[1], linestyle=:solid)
lines!(ax2, 1:length(pca_vars_large), cumsum(pca_vars_large)./sum(pca_vars_large), linewidth=3.5, color=mints_colors[1], linestyle=:dot)
lines!(ax2, 1:length(pca_vars_small), cumsum(pca_vars_small)./sum(pca_vars_small), linewidth=2, color=mints_colors[1], linestyle=:dash)

xlims!(ax2, 1, 10)
fig

save(joinpath(figpath, "pca-explained-variance.png"), fig)
save(joinpath(figpath, "pca-explained-variance.pdf"), fig)


# ----------------------------------------------------------------------
# --------------- Linear Unmixing on "large" Dataset -------------------
# ----------------------------------------------------------------------

savepath = joinpath(figpath, "large")
@assert ispath(savepath)

n_nodes = 50_000
n_rbfs = 25_000
s = 0.05
λ = 0.1
Nᵥ = 6

gsm = GSMBig(n_nodes=n_nodes, n_rbfs=n_rbfs, Nv=Nᵥ, s=s, λ=λ, nonlinear=false, linear=true, bias=false, make_positive=true, tol=1e-6, nepochs=500, rand_init=false, rng=StableRNG(42))
mach = machine(gsm, df_large)
fit!(mach, verbosity=1)

abund = DataFrame(MLJ.transform(mach, df_large));

rpt = report(mach)
node_means = rpt[:node_data_means]
Q = rpt[:Q]
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]
stdev = sqrt(rpt[:β⁻¹])



# visualize abundances so we can see if there are any useless dimensions
fig = Figure();
ax = Axis(fig[1,1], xlabel="Endmember", ylabel="log10 Abundance", xticks=(1:15, ["z₁", "z₂", "z₃", "z₄", "z₅", "z₆", "z₇", "z₈", "z₉", "z₁₀", "z₁₁", "z₁₂", "z₁₃", "z₁₄", "z₁₅"]));
for i ∈ 1:size(abund, 2)
    cat = i*ones(Int, size(abund, 1))
    bp = boxplot!(ax, cat, log10.(abund[:,i]))
    # bp = boxplot!(ax, cat, abund[:,i])
    # bp = violin!(ax, cat, abund[:,i], show_median=true)
end
ylims!(ax, -5, 0.1)
fig

save(joinpath(savepath, "extra-endmembers-boxplot.png"), fig)
save(joinpath(savepath, "extra-endmembers-boxplot.pdf"), fig)






# plot log likelihoods
fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Log-likelihood", yscale=log10)
lines!(ax, 2:length(llhs), llhs[2:end], linewidth=3)
fig

save(joinpath(savepath, "llh.png"), fig)
save(joinpath(savepath, "llh.pdf"), fig)


fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Q (a.u.)", yscale=log10)
lines!(ax, 2:length(Q), Q[2:end], linewidth=3)
fig

save(joinpath(savepath, "Q-fit.png"), fig)
save(joinpath(savepath, "Q-fit.pdf"), fig)

# plot endmembers
fig = Figure(; resolution=(960, 540));
# fig = Figure(; resolution=(15_000, 540));
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance");
ax2 = Axis(fig[1,2], xlabel="λ", yticksvisible=false, yticklabelsvisible=false);
linkyaxes!(ax, ax2)

Ncolors = length(mints_colors)

i = 1
for idx ∈ idx_vertices
    Rout = node_means[:,idx]
    band!(ax, λs, Rout .- (2*stdev), Rout .+ (2*stdev), color=(mints_colors[i%Ncolors+1], 0.5), transparency=true)
    li = lines!(ax, λs, Rout, color=mints_colors[i%Ncolors+1], linewidth=2)
    i += 1
end

names(df_ref)

b1 = lines!(ax2, df_ref.λ, df_ref.bitumen, color=:black, linewidth=2)
b2 = lines!(ax2, df_svc.λ, df_svc.bitumen, color=:black, linewidth=2, linestyle=:dash)

rm1 = lines!(ax2, df_ref.λ, df_ref.red_metal, color=:red, linewidth=2)
rm2 = lines!(ax2, df_svc.λ, df_svc.red_metal, color=:red, linewidth=2, linestyle=:dash)

bf1 = lines!(ax2, df_ref.λ, df_ref.blue_fabric, color=:blue, linewidth=2)
bf2 = lines!(ax2, df_svc.λ, df_svc.blue_fabric, color=:blue, linewidth=2, linestyle=:dash)

rf1 = lines!(ax2, df_ref.λ, df_ref.red_fabric, color=mints_colors[2], linewidth=2)
rf2 = lines!(ax2, df_svc.λ, df_svc.red_fabric, color=mints_colors[2], linewidth=2, linestyle=:dash)

gf1 = lines!(ax2, df_ref.λ, df_ref.green_fabric, color=mints_colors[1], linewidth=2)
gf2 = lines!(ax2, df_svc.λ, df_svc.green_fabric, color=mints_colors[1], linewidth=2, linestyle=:dash)


gs1 = lines!(ax2, df_ref.λ, df_ref.grass, color=:green, linewidth=2)
gs2 = lines!(ax2, df_svc.λ, df_svc.grass, color=:green, linewidth=2, linestyle=:dash)

xlims!(ax, λs[1], λs[end])
xlims!(ax2, df_ref.λ[1], df_ref.λ[end])
ylims!(ax, 0, 0.9)

fig[1,3] = Legend(fig, [b1, rm1, bf1, rf1, gf1, gs1], ["Bitumen", "Red Metal", "Blue Fabric", "Red Fabric", "Green Fabric", "Grass"], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=13, height=-5)

fig


save(joinpath(savepath, "extracted-endmembers.png"), fig)
save(joinpath(savepath, "extracted-endmembers.pdf"), fig)



# compute minimum spectral angle

function spectral_angle(r1, r2)
    return acos(dot(r1, r2)/(norm(r1) * norm(r2)))
end


# bitumen
res = Dict(
    "Bitumen" => minimum([spectral_angle(node_means[:,i], df_ref.bitumen) for i ∈ 1:Nᵥ]),
    "Red Metal" => minimum([spectral_angle(node_means[:,i], df_ref.red_metal) for i ∈ 1:Nᵥ]),
    "Blue Fabric" => minimum([spectral_angle(node_means[:,i], df_ref.blue_fabric) for i ∈ 1:Nᵥ]),
    "Red Fabric" => minimum([spectral_angle(node_means[:,i], df_ref.red_fabric) for i ∈ 1:Nᵥ]),
    "Green Fabric" => minimum([spectral_angle(node_means[:,i], df_ref.green_fabric) for i ∈ 1:Nᵥ]),
    "Grass" => minimum([spectral_angle(node_means[:,i], df_ref.grass) for i ∈ 1:Nᵥ]),
)
res["Total"] = sum(res[key] for key ∈ keys(res))


# save dict to JSON file
open(joinpath(savepath, "spectral-angle_N=$(Nᵥ).json"), "w") do f
    JSON.print(f, res)
end




# maybe a radial plot with abundances on on spokes?





# ----------------------------------------------------------------------
# --------------- Linear Unmixing on "small" Dataset -------------------
# ----------------------------------------------------------------------

savepath = joinpath(figpath, "small")
@assert ispath(savepath)

n_nodes = 50_000
n_rbfs = 10
λ = 0.1
Nᵥ = 6

gsm = GSMBig(n_nodes=n_nodes, n_rbfs=n_rbfs, Nv=Nᵥ, λ=λ, nonlinear=false, linear=true, bias=false, make_positive=true, tol=1e-6, nepochs=500, rand_init=false, rng=StableRNG(42))
mach = machine(gsm, df_small)
fit!(mach, verbosity=1)

abund = DataFrame(MLJ.transform(mach, df_small));

rpt = report(mach)
node_means = rpt[:node_data_means]
Q = rpt[:Q]
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]
stdev = sqrt(rpt[:β⁻¹])



# visualize abundances so we can see if there are any useless dimensions
fig = Figure();
ax = Axis(fig[1,1], xlabel="Endmember", ylabel="log10 Abundance", xticks=(1:15, ["z₁", "z₂", "z₃", "z₄", "z₅", "z₆", "z₇", "z₈", "z₉", "z₁₀", "z₁₁", "z₁₂", "z₁₃", "z₁₄", "z₁₅"]));
for i ∈ 1:size(abund, 2)
    cat = i*ones(Int, size(abund, 1))
    bp = boxplot!(ax, cat, log10.(abund[:,i]))
end
ylims!(ax, -5, 0.1)
fig

save(joinpath(savepath, "extra-endmembers-boxplot.png"), fig)
save(joinpath(savepath, "extra-endmembers-boxplot.pdf"), fig)


# plot log likelihoods
fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Log-likelihood", yscale=log10)
lines!(ax, 2:length(llhs), llhs[2:end], linewidth=3)
fig

save(joinpath(savepath, "llh.png"), fig)
save(joinpath(savepath, "llh.pdf"), fig)


fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Q (a.u.)", yscale=log10)
lines!(ax, 2:length(Q), Q[2:end], linewidth=3)
fig

save(joinpath(savepath, "Q-fit.png"), fig)
save(joinpath(savepath, "Q-fit.pdf"), fig)

# plot endmembers
fig = Figure(; resolution=(960, 540));
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance");
ax2 = Axis(fig[1,2], xlabel="λ", yticksvisible=false, yticklabelsvisible=false);
linkyaxes!(ax, ax2)

Ncolors = length(mints_colors)

i = 1
for idx ∈ idx_vertices
    Rout = node_means[:,idx]
    band!(ax, λs, Rout .- (2*stdev), Rout .+ (2*stdev), color=(mints_colors[i%Ncolors+1], 0.5), transparency=true)
    li = lines!(ax, λs, Rout, color=mints_colors[i%Ncolors+1], linewidth=2)
    i += 1
end

names(df_ref)

b1 = lines!(ax2, df_ref.λ, df_ref.bitumen, color=:black, linewidth=2)
b2 = lines!(ax2, df_svc.λ, df_svc.bitumen, color=:black, linewidth=2, linestyle=:dash)

rm1 = lines!(ax2, df_ref.λ, df_ref.red_metal, color=:red, linewidth=2)
rm2 = lines!(ax2, df_svc.λ, df_svc.red_metal, color=:red, linewidth=2, linestyle=:dash)

bf1 = lines!(ax2, df_ref.λ, df_ref.blue_fabric, color=:blue, linewidth=2)
bf2 = lines!(ax2, df_svc.λ, df_svc.blue_fabric, color=:blue, linewidth=2, linestyle=:dash)

rf1 = lines!(ax2, df_ref.λ, df_ref.red_fabric, color=mints_colors[2], linewidth=2)
rf2 = lines!(ax2, df_svc.λ, df_svc.red_fabric, color=mints_colors[2], linewidth=2, linestyle=:dash)

gf1 = lines!(ax2, df_ref.λ, df_ref.green_fabric, color=mints_colors[1], linewidth=2)
gf2 = lines!(ax2, df_svc.λ, df_svc.green_fabric, color=mints_colors[1], linewidth=2, linestyle=:dash)


gs1 = lines!(ax2, df_ref.λ, df_ref.grass, color=:green, linewidth=2)
gs2 = lines!(ax2, df_svc.λ, df_svc.grass, color=:green, linewidth=2, linestyle=:dash)

xlims!(ax, λs[1], λs[end])
xlims!(ax2, df_ref.λ[1], df_ref.λ[end])
ylims!(ax, 0, 0.9)

fig[1,3] = Legend(fig, [b1, rm1, bf1, rf1, gf1, gs1], ["Bitumen", "Red Metal", "Blue Fabric", "Red Fabric", "Green Fabric", "Grass"], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=13, height=-5)

fig


save(joinpath(savepath, "extracted-endmembers.png"), fig)
save(joinpath(savepath, "extracted-endmembers.pdf"), fig)


# bitumen
res = Dict(
    "Bitumen" => minimum([spectral_angle(node_means[:,i], df_ref.bitumen) for i ∈ 1:Nᵥ]),
    "Red Metal" => minimum([spectral_angle(node_means[:,i], df_ref.red_metal) for i ∈ 1:Nᵥ]),
    "Blue Fabric" => minimum([spectral_angle(node_means[:,i], df_ref.blue_fabric) for i ∈ 1:Nᵥ]),
    "Red Fabric" => minimum([spectral_angle(node_means[:,i], df_ref.red_fabric) for i ∈ 1:Nᵥ]),
    "Green Fabric" => minimum([spectral_angle(node_means[:,i], df_ref.green_fabric) for i ∈ 1:Nᵥ]),
    "Grass" => minimum([spectral_angle(node_means[:,i], df_ref.grass) for i ∈ 1:Nᵥ]),
)
res["Total"] = sum(res[key] for key ∈ keys(res))


# save dict to JSON file
open(joinpath(savepath, "spectral-angle_N=$(Nᵥ).json"), "w") do f
    JSON.print(f, res)
end






# ----------------------------------------------------------------------
# --------------- Linear Unmixing on "all" Dataset -------------------
# ----------------------------------------------------------------------

savepath = joinpath(figpath, "all")
@assert ispath(savepath)


n_nodes = 50_000
n_rbfs = 10
λ = 0.1     # 1.32985
# λ = 0.0001  # 1.33595
# λ = 10      # 1.43143
# λ = 0.01    # 1.33292
# λ = 1e-5
Nᵥ = 7

gsm
gsm = GSMBig(n_nodes=n_nodes, n_rbfs=n_rbfs, Nv=Nᵥ, λ=λ, nonlinear=false, linear=true, bias=false, make_positive=true, tol=1e-6, nepochs=500, rand_init=false, rng=StableRNG(42))
mach = machine(gsm, df_all)
fit!(mach, verbosity=1)

abund = DataFrame(MLJ.transform(mach, df_all));

rpt = report(mach)
node_means = rpt[:node_data_means]
Q = rpt[:Q]
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]
stdev = sqrt(rpt[:β⁻¹])



# visualize abundances so we can see if there are any useless dimensions
fig = Figure();
ax = Axis(fig[1,1], xlabel="Endmember", ylabel="log10 Abundance", xticks=(1:15, ["z₁", "z₂", "z₃", "z₄", "z₅", "z₆", "z₇", "z₈", "z₉", "z₁₀", "z₁₁", "z₁₂", "z₁₃", "z₁₄", "z₁₅"]));
for i ∈ 1:size(abund, 2)
    cat = i*ones(Int, size(abund, 1))
    bp = boxplot!(ax, cat, log10.(abund[:,i]))
end
ylims!(ax, -5, 0.1)
fig

save(joinpath(savepath, "extra-endmembers-boxplot.png"), fig)
save(joinpath(savepath, "extra-endmembers-boxplot.pdf"), fig)


# plot log likelihoods
fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Log-likelihood", yscale=log10)
lines!(ax, 2:length(llhs), llhs[2:end], linewidth=3)
fig

save(joinpath(savepath, "llh.png"), fig)
save(joinpath(savepath, "llh.pdf"), fig)


fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Q (a.u.)", yscale=log10)
lines!(ax, 2:length(Q), Q[2:end], linewidth=3)
fig

save(joinpath(savepath, "Q-fit.png"), fig)
save(joinpath(savepath, "Q-fit.pdf"), fig)

# plot endmembers
fig = Figure(; resolution=(960, 540));
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance");
ax2 = Axis(fig[1,2], xlabel="λ", yticksvisible=false, yticklabelsvisible=false);
linkyaxes!(ax, ax2)

Ncolors = length(mints_colors)

i = 1
for idx ∈ idx_vertices
    Rout = node_means[:,idx]
    band!(ax, λs, Rout .- (2*stdev), Rout .+ (2*stdev), color=(mints_colors[i%Ncolors+1], 0.5), transparency=true)
    li = lines!(ax, λs, Rout, color=mints_colors[i%Ncolors+1], linewidth=2)
    i += 1
end


b1 = lines!(ax2, df_ref.λ, df_ref.bitumen, color=:black, linewidth=2)
b2 = lines!(ax2, df_svc.λ, df_svc.bitumen, color=:black, linewidth=2, linestyle=:dash)

rm1 = lines!(ax2, df_ref.λ, df_ref.red_metal, color=:red, linewidth=2)
rm2 = lines!(ax2, df_svc.λ, df_svc.red_metal, color=:red, linewidth=2, linestyle=:dash)

bf1 = lines!(ax2, df_ref.λ, df_ref.blue_fabric, color=:blue, linewidth=2)
bf2 = lines!(ax2, df_svc.λ, df_svc.blue_fabric, color=:blue, linewidth=2, linestyle=:dash)

rf1 = lines!(ax2, df_ref.λ, df_ref.red_fabric, color=mints_colors[2], linewidth=2)
rf2 = lines!(ax2, df_svc.λ, df_svc.red_fabric, color=mints_colors[2], linewidth=2, linestyle=:dash)

gf1 = lines!(ax2, df_ref.λ, df_ref.green_fabric, color=mints_colors[1], linewidth=2)
gf2 = lines!(ax2, df_svc.λ, df_svc.green_fabric, color=mints_colors[1], linewidth=2, linestyle=:dash)


gs1 = lines!(ax2, df_ref.λ, df_ref.grass, color=:green, linewidth=2)
gs2 = lines!(ax2, df_svc.λ, df_svc.grass, color=:green, linewidth=2, linestyle=:dash)

xlims!(ax, λs[1], λs[end])
xlims!(ax2, df_ref.λ[1], df_ref.λ[end])
ylims!(ax, 0, 0.9)

fig[1,3] = Legend(fig, [b1, rm1, bf1, rf1, gf1, gs1], ["Bitumen", "Red Metal", "Blue Fabric", "Red Fabric", "Green Fabric", "Grass"], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=13, height=-5)

fig


save(joinpath(savepath, "extracted-endmembers.png"), fig)
save(joinpath(savepath, "extracted-endmembers.pdf"), fig)


# bitumen
res = Dict(
    "Bitumen" => minimum([spectral_angle(node_means[:,i], df_ref.bitumen) for i ∈ 1:Nᵥ]),
    "Red Metal" => minimum([spectral_angle(node_means[:,i], df_ref.red_metal) for i ∈ 1:Nᵥ]),
    "Blue Fabric" => minimum([spectral_angle(node_means[:,i], df_ref.blue_fabric) for i ∈ 1:Nᵥ]),
    "Red Fabric" => minimum([spectral_angle(node_means[:,i], df_ref.red_fabric) for i ∈ 1:Nᵥ]),
    "Green Fabric" => minimum([spectral_angle(node_means[:,i], df_ref.green_fabric) for i ∈ 1:Nᵥ]),
    "Grass" => minimum([spectral_angle(node_means[:,i], df_ref.grass) for i ∈ 1:Nᵥ]),
)
res["Total"] = sum(res[key] for key ∈ keys(res))


# save dict to JSON file
open(joinpath(savepath, "spectral-angle_N=$(Nᵥ).json"), "w") do f
    JSON.print(f, res)
end






# ----------------------------------------------------------------------
# --------------- Nonlinear Unmixing on "all" Dataset -------------------
# ----------------------------------------------------------------------

savepath = joinpath(figpath, "all-nonlinear")
if !ispath(savepath)
    mkpath(savepath)
end


n_nodes = 25_000
n_rbfs = 5_000
s = 0.01
λ = 0.1
Nᵥ = 10


gsm = GSMBig(n_nodes=n_nodes, n_rbfs=n_rbfs, Nv=Nᵥ, λ=λ, nonlinear=true, linear=true, bias=false, make_positive=true, tol=1e-6, nepochs=500, rand_init=false, rng=StableRNG(42))
mach = machine(gsm, df_all)
fit!(mach, verbosity=1)

abund = DataFrame(MLJ.transform(mach, df_all));

size(abund)
findall(abund[:,6] .< 0)


rpt = report(mach)
node_means = rpt[:node_data_means]
Q = rpt[:Q]
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]
stdev = sqrt(rpt[:β⁻¹])



# visualize abundances so we can see if there are any useless dimensions
fig = Figure();
ax = Axis(fig[1,1], xlabel="Endmember", ylabel="log10 Abundance", xticks=(1:15, ["z₁", "z₂", "z₃", "z₄", "z₅", "z₆", "z₇", "z₈", "z₉", "z₁₀", "z₁₁", "z₁₂", "z₁₃", "z₁₄", "z₁₅"]));
for i ∈ 1:size(abund, 2)
    cat = i*ones(Int, size(abund, 1))
    bp = boxplot!(ax, cat, log10.(abund[:,i]))
end
ylims!(ax, -5, 0.1)
fig

save(joinpath(savepath, "extra-endmembers-boxplot.png"), fig)
save(joinpath(savepath, "extra-endmembers-boxplot.pdf"), fig)


# plot log likelihoods
fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Log-likelihood", yscale=log10)
lines!(ax, 2:length(llhs), llhs[2:end], linewidth=3)
fig

save(joinpath(savepath, "llh.png"), fig)
save(joinpath(savepath, "llh.pdf"), fig)


fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Q (a.u.)", yscale=log10)
lines!(ax, 2:length(Q), Q[2:end], linewidth=3)
fig

save(joinpath(savepath, "Q-fit.png"), fig)
save(joinpath(savepath, "Q-fit.pdf"), fig)

# plot endmembers
fig = Figure(; resolution=(960, 540));
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance");
ax2 = Axis(fig[1,2], xlabel="λ", yticksvisible=false, yticklabelsvisible=false);
linkyaxes!(ax, ax2)

Ncolors = length(mints_colors)

i = 1
for idx ∈ idx_vertices
    Rout = node_means[:,idx]
    band!(ax, λs, Rout .- (2*stdev), Rout .+ (2*stdev), color=(mints_colors[i%Ncolors+1], 0.5), transparency=true)
    li = lines!(ax, λs, Rout, color=mints_colors[i%Ncolors+1], linewidth=2)
    i += 1
end


b1 = lines!(ax2, df_ref.λ, df_ref.bitumen, color=:black, linewidth=2)
b2 = lines!(ax2, df_svc.λ, df_svc.bitumen, color=:black, linewidth=2, linestyle=:dash)

rm1 = lines!(ax2, df_ref.λ, df_ref.red_metal, color=:red, linewidth=2)
rm2 = lines!(ax2, df_svc.λ, df_svc.red_metal, color=:red, linewidth=2, linestyle=:dash)

bf1 = lines!(ax2, df_ref.λ, df_ref.blue_fabric, color=:blue, linewidth=2)
bf2 = lines!(ax2, df_svc.λ, df_svc.blue_fabric, color=:blue, linewidth=2, linestyle=:dash)

rf1 = lines!(ax2, df_ref.λ, df_ref.red_fabric, color=mints_colors[2], linewidth=2)
rf2 = lines!(ax2, df_svc.λ, df_svc.red_fabric, color=mints_colors[2], linewidth=2, linestyle=:dash)

gf1 = lines!(ax2, df_ref.λ, df_ref.green_fabric, color=mints_colors[1], linewidth=2)
gf2 = lines!(ax2, df_svc.λ, df_svc.green_fabric, color=mints_colors[1], linewidth=2, linestyle=:dash)


gs1 = lines!(ax2, df_ref.λ, df_ref.grass, color=:green, linewidth=2)
gs2 = lines!(ax2, df_svc.λ, df_svc.grass, color=:green, linewidth=2, linestyle=:dash)

xlims!(ax, λs[1], λs[end])
xlims!(ax2, df_ref.λ[1], df_ref.λ[end])
ylims!(ax, 0, 0.9)

fig[1,3] = Legend(fig, [b1, rm1, bf1, rf1, gf1, gs1], ["Bitumen", "Red Metal", "Blue Fabric", "Red Fabric", "Green Fabric", "Grass"], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=13, height=-5)

fig


save(joinpath(savepath, "extracted-endmembers.png"), fig)
save(joinpath(savepath, "extracted-endmembers.pdf"), fig)


# bitumen
res = Dict(
    "Bitumen" => minimum([spectral_angle(node_means[:,i], df_ref.bitumen) for i ∈ 1:Nᵥ]),
    "Red Metal" => minimum([spectral_angle(node_means[:,i], df_ref.red_metal) for i ∈ 1:Nᵥ]),
    "Blue Fabric" => minimum([spectral_angle(node_means[:,i], df_ref.blue_fabric) for i ∈ 1:Nᵥ]),
    "Red Fabric" => minimum([spectral_angle(node_means[:,i], df_ref.red_fabric) for i ∈ 1:Nᵥ]),
    "Green Fabric" => minimum([spectral_angle(node_means[:,i], df_ref.green_fabric) for i ∈ 1:Nᵥ]),
    "Grass" => minimum([spectral_angle(node_means[:,i], df_ref.grass) for i ∈ 1:Nᵥ]),
)
res["Total"] = sum(res[key] for key ∈ keys(res))


# save dict to JSON file
open(joinpath(savepath, "spectral-angle_N=$(Nᵥ).json"), "w") do f
    JSON.print(f, res)
end







