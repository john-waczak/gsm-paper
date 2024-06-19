using CairoMakie
using MLJ, GenerativeTopographicMapping
using DataFrames, CSV
using TernaryDiagrams
using Distributions
using StableRNGs
using Statistics, LinearAlgebra
using Random



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


savepath = "./figures/synthetic-usgs"
if !ispath(savepath)
    mkpath(savepath)
end


# Generate synthetic datasets
df = CSV.read("./data/usgs/usgs.csv", DataFrame)

names(df)

λs = df.λ
names(df)
min_to_use = ["Carnallite", "Ammonium Illite", "Biotite"]  # based on original VCA paper
# min_to_use = ["Carnallite", "Ammonium Illite", "Rutile"]  # based on original VCA paper
R1 = df[:, min_to_use[1]]
R2 = df[:, min_to_use[2]]
R3 = df[:, min_to_use[3]]



Npoints = 1_000
α_true = [1/3, 1/3, 1/3]
f_dir = Dirichlet(α_true)
abund = rand(rng, f_dir, Npoints)


X = zeros(Npoints, length(λs));
σnoise = 0.045

for i ∈ axes(X, 1)
    X[i,:] .= (abund[1,i] .* R1) .+ (abund[2,i] .* R2) .+ (abund[3,i] .* R3)
    X[i,:] .= X[i,:] .+ rand(rng, Normal(0, σnoise), length(λs))
end

colnames = Symbol.(["λ_" * lpad(i, 3, "0") for i ∈ 1:length(λs)])
X = DataFrame(X, colnames);

Xnl = zeros(Npoints, length(λs));
σnoise_nl = 0.02
γ = 0.5

for i ∈ axes(X, 1)

    a = abund[1,i]
    b = abund[2,i]
    c = abund[3,i]

    d = γ * a * b
    e = γ * a * c
    f = γ * b * c

    Xnl[i,:] .= (a .* R1) .+ (b .* R2) .+ (c .* R3) .+ (d .* R1 .* R2) .+ (e .* R1 .* R3) .+ (f .* R2 .* R3)
    Xnl[i,:] .= Xnl[i,:] .+ rand(rng, Normal(0, σnoise_nl), length(λs))
end

colnames = Symbol.(["λ_" * lpad(i, 3, "0") for i ∈ 1:length(λs)])
Xnl = DataFrame(Xnl, colnames);




# ---------------------------------------------
# ------------- LINEAR MODEL -----------------
# ---------------------------------------------

figpath = joinpath(savepath, "linear-uniform")
if !ispath(figpath)
    mkpath(figpath)
end

# visualize the distribution of abundances
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
save(joinpath(figpath, "abundance-orig.pdf"), fig)

fig


# visualize the data
fig = Figure();
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance")
for i ∈ 1:10:nrow(X)
    Rᵢ = Array(X[i,:])
    lines!(ax, λs, Rᵢ, color=CairoMakie.RGBAf(abund[:,i]..., 0.25), linewidth=1)
end
xlims!(ax, λs[1], λs[end])
fig

save(joinpath(figpath, "sample-spectra.png"), fig)
save(joinpath(figpath, "sample-spectra.pdf"), fig)


# fit linear GSM
k = 75
m = 10
λ = 0.001
Nᵥ = 3

gsm_l = GSM(k=k, m=m, Nv=Nᵥ, λ=λ,  nonlinear=false, linear=true, bias=false, make_positive=true, tol=1e-9, nepochs=100, rand_init=false, rng=StableRNG(42))
mach_l = machine(gsm_l, X)
fit!(mach_l, verbosity=1)

abund_l = DataFrame(MLJ.transform(mach_l, X));

model = fitted_params(mach_l)[:gsm]
rpt = report(mach_l)
node_means = rpt[:node_data_means]
Q = rpt[:Q]
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]
stdev = sqrt(rpt[:β⁻¹])  # 0.04489901084606633

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



# plot endmembers for linear data
fig = Figure();
ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");

l1 = lines!(ax, λs, R1, color=mints_colors[2], linewidth=3)
l2 = lines!(ax, λs, R2, color=mints_colors[1], linewidth=3)
l3 = lines!(ax, λs, R3, color=mints_colors[3], linewidth=3)

ls_fit = []
linestyles = [:solid, :dash, :dot]
i = 1
for idx ∈ idx_vertices
    Rout = node_means[:,idx]
    band!(ax, λs, Rout .- (2*stdev), Rout .+ (2*stdev), color=(:gray, 0.35))
    li = lines!(ax, λs, Rout, color=:gray, linewidth=2, linestyle=linestyles[i])
    push!(ls_fit, li)
    i += 1
end

fig[1,1] = Legend(fig, [l1, l2, l3, ls_fit...], [min_to_use..., "Vertex 1", "Vertex 2", "Vertex 3"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
xlims!(ax, λs[1], λs[end])

save(joinpath(figpath, "extracted-endmembers.png"), fig)
save(joinpath(figpath, "extracted-endmembers.pdf"), fig)

fig



# so v1 -> blue, v2 -> green, v3 -> red

# evaluate performance using spectral angle
function spectral_angle(r1, r2)
    return acos(dot(r1, r2)/(norm(r1) * norm(r2)))
end

function rmse(r1, r2)
    return sqrt(sum((r1 .- r2).^2)/length(r1))
end

function spectral_information_divergence(r1, r2)
    p = r1 ./ sum(r1)
    q = r2 ./ sum(r2)

    return sum(p .* log.(p ./ q)) + sum(q .* log.(q ./ p))
end


# Evaluate Endmember accuracy

θ1 = spectral_angle(node_means[:, idx_vertices[1]], R3)
θ2 = spectral_angle(node_means[:, idx_vertices[2]], R2)
θ3 = spectral_angle(node_means[:, idx_vertices[3]], R1)


# 0.02268606690205627
# 0.005060356035547636
# 0.011310280096455266


rmse1 = rmse(node_means[:, idx_vertices[1]], R3)
rmse2 = rmse(node_means[:, idx_vertices[2]], R2)
rmse3 = rmse(node_means[:, idx_vertices[3]], R1)

# 0.006372666002815531
# 0.004133218515423938
# 0.007070547363184486

sid1 = spectral_information_divergence(node_means[:, idx_vertices[1]], R3)
sid2 = spectral_information_divergence(node_means[:, idx_vertices[2]], R2)
sid3 = spectral_information_divergence(node_means[:, idx_vertices[3]], R1)

# 0.0008944870569942626
# 3.3577485052296784e-5
# 0.0006132242550569524

# Evaluate abundance accuracy
a_rmse_1 = rmse(abund[3,:], abund_l.Z1)
a_rmse_1 = rmse(abund[2,:], abund_l.Z2)
a_rmse_1 = rmse(abund[1,:], abund_l.Z3)

# 0.007735307757786526
# 0.017657189916274397
# 0.019090156911127594





# DOES THIS EVEN MAKE SENSE???

# ---------------------------------------------
# ----------- NONLINEAR MODEL -----------------
# ---------------------------------------------

figpath = joinpath(savepath, "nonlinear")
if !ispath(figpath)
    mkpath(figpath)
end

# visualize the distribution of abundances
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
save(joinpath(figpath, "abundance-orig.pdf"), fig)

fig


# visualize the data
fig = Figure();
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance")
for i ∈ 1:10:nrow(X)
    Rᵢ = Array(X[i,:])
    lines!(ax, λs, Rᵢ, color=CairoMakie.RGBAf(abund[:,i]..., 0.25), linewidth=1)
end
xlims!(ax, λs[1], λs[end])
fig

save(joinpath(figpath, "sample-spectra.png"), fig)
save(joinpath(figpath, "sample-spectra.pdf"), fig)


# fit nonlinear GSM
k = 75
m = 15
λ = 0.01
Nᵥ = 3

gsm_l = GSM(k=k, m=m, Nv=Nᵥ, λ=λ,  nonlinear=true, linear=true, bias=false, make_positive=false, tol=1e-5, nepochs=100, rand_init=true, rng=StableRNG(42))
mach_l = machine(gsm_l, Xnl)
fit!(mach_l, verbosity=1)


abund_l = DataFrame(MLJ.transform(mach_l, Xnl));

model = fitted_params(mach_l)[:gsm]
rpt = report(mach_l)
node_means = rpt[:node_data_means]
Q = rpt[:Q]
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]
stdev = sqrt(rpt[:β⁻¹])  # 0.04489901084606633

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



# plot endmembers for linear data
fig = Figure();
ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");

l1 = lines!(ax, λs, R1, color=mints_colors[2], linewidth=3)
l2 = lines!(ax, λs, R2, color=mints_colors[1], linewidth=3)
l3 = lines!(ax, λs, R3, color=mints_colors[3], linewidth=3)

ls_fit = []
linestyles = [:solid, :dash, :dot]
i = 1
for idx ∈ idx_vertices
    Rout = node_means[:,idx]
    band!(ax, λs, Rout .- (2*stdev), Rout .+ (2*stdev), color=(:gray, 0.35))
    li = lines!(ax, λs, Rout, color=:gray, linewidth=2, linestyle=linestyles[i])
    push!(ls_fit, li)
    i += 1
end

fig[1,1] = Legend(fig, [l1, l2, l3, ls_fit...], [min_to_use..., "Vertex 1", "Vertex 2", "Vertex 3"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
xlims!(ax, λs[1], λs[end])

save(joinpath(figpath, "extracted-endmembers.png"), fig)
save(joinpath(figpath, "extracted-endmembers.pdf"), fig)


fig



