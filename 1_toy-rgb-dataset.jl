using CairoMakie
using MLJ, GenerativeTopographicMapping
using DataFrames
using TernaryDiagrams
using Distributions
using StableRNGs
using Statistics
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


savepath = "./figures/synthetic-rgb"
if !ispath(savepath)
    mkpath(savepath)
end




# Generate toy dataset of "RGB" spectra
λs = range(350, stop=750, length=1000)

Rb = exp.(-(λs .- 460.0).^2 ./(2*(22)^2))
Rg = exp.(-(λs .- 525.0).^2 ./(2*(28)^2))
Rr = exp.(-(λs .- 625.0).^2 ./(2*(20)^2))

Rb .= Rb./maximum(Rb)
Rg .= Rg./maximum(Rg)
Rr .= Rr./maximum(Rr)

fig = Figure();
ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");
lr = lines!(ax, λs, Rr, color=mints_colors[2], linewidth=2)
lg = lines!(ax, λs, Rg, color=mints_colors[1], linewidth=2)
lb = lines!(ax, λs, Rb, color=mints_colors[3], linewidth=2)
fig[1,1] = Legend(fig, [lr, lg, lb], ["Red Band", "Green Band", "Blue Band"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
xlims!(ax, λs[1], λs[end])

fig

save(joinpath(savepath, "rgb-orig.png"), fig)
save(joinpath(savepath, "rgb-orig.pdf"), fig)



# generate dataframe of reflectance values from combined dataset

# Npoints = 10_000
Npoints = 25_000
α_true = 0.05 * ones(3)
f_dir = Dirichlet(α_true)
abund = rand(rng, f_dir, Npoints)

α_true_2 = [0.25, 1.1, 2.25]
f_dir_2 = Dirichlet(α_true_2)
abund_2 = rand(rng, f_dir_2, Npoints)


X = zeros(Npoints, length(λs));       # linear mixing
Xnoise = zeros(Npoints, length(λs));  #
X2 = zeros(Npoints, length(λs));      # linear mixing w/ asymmetric sampling
Xnl = zeros(Npoints, length(λs));      # non-linear mixing

for i ∈ axes(X, 1)
    X[i,:] .= (abund[1,i] .* Rr) .+ (abund[2,i] .* Rg) .+ (abund[3,i] .* Rb)
    Xnoise[i,:] .= X[i,:] .+ rand(rng, Normal(0, 0.01), length(λs))
    X2[i,:] .= (abund_2[1,i] .* Rr) .+ (abund_2[2,i] .* Rg) .+ (abund_2[3,i] .* Rb)

    Xnl[i,:] .= (abund[1,i] .* Rr) .+ (abund[2,i] .* Rg) .+ (abund[3,i] .* Rb) .+ (abund[1,i] * abund[2,i] .* Rr .* Rg) .+ (abund[1,i] * abund[3,i] .* Rr .* Rb)  .+ (abund[2,i] * abund[3,i] .* Rg .* Rb)
end

names = Symbol.(["λ_" * lpad(i, 3, "0") for i ∈ 1:length(λs)])

X = DataFrame(X, names);
Xnoise = DataFrame(Xnoise, names);
X2 = DataFrame(X2, names);
Xnl = DataFrame(Xnl, names);




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

text!(ax, Point2f(0.07, 0.5), text="Blue", fontsize=22)
text!(ax, Point2f(0.825, 0.5), text="Green", fontsize=22)
text!(ax, Point2f(0.45, -0.175), text="Red", fontsize=22)

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


# demonstrate dimensionality identification via sparsity of latent space axes
k = 10
m = 5
λ = 0.001
Nᵥ = 5

# Do inital fit with Nᵥ > Nᵥ_true to test dimensionality identification
gsm_l = GSM(k=k, m=m, Nv=Nᵥ, λ=λ,  nonlinear=false, linear=true, bias=false, make_positive=true, tol=1e-5, nepochs=25, rand_init=false, rng=StableRNG(42))
mach_l = machine(gsm_l, X)
fit!(mach_l, verbosity=1)
abund_l = DataFrame(MLJ.transform(mach_l, X));

fig = Figure();
ax = Axis(fig[1,1], xlabel="Endmember", ylabel="Abundance", xticks=(1:5, ["z₁", "z₂", "z₃", "z₄", "z₅"]));
for i ∈ 1:size(abund_l, 2)
    cat = i*ones(Int, size(abund_l, 1))
    bp = boxplot!(ax, cat, abund_l[:,i])
end
fig

save(joinpath(figpath, "extra-endmembers-boxplot.png"), fig)
save(joinpath(figpath, "extra-endmembers-boxplot.pdf"), fig)



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
stdev = sqrt(rpt[:β⁻¹])


model = fitted_params(mach_l)[:gsm]
πk = model.πk
Z = model.Z

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

lr = lines!(ax, λs, Rr, color=mints_colors[2], linewidth=3)
lg = lines!(ax, λs, Rg, color=mints_colors[1], linewidth=3)
lb = lines!(ax, λs, Rb, color=mints_colors[3], linewidth=3)

ls_fit = []
linestyles = [:solid, :dash, :dot]
i = 1
for idx ∈ idx_vertices
    Rout = node_means[:,idx]
    band!(ax, λs, Rout .- (2*stdev), Rout .+ (2*stdev), color=(:gray, 0.5))
    li = lines!(ax, λs, Rout, color=:gray, linewidth=2, linestyle=linestyles[i])
    push!(ls_fit, li)
    i += 1
end

fig[1,1] = Legend(fig, [lr, lg, lb, ls_fit...], ["Red Band", "Green Band", "Blue Band", "Vertex 1", "Vertex 2", "Vertex 3"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
xlims!(ax, λs[1], λs[end])

save(joinpath(figpath, "extracted-endmembers.png"), fig)
save(joinpath(figpath, "extracted-endmembers.pdf"), fig)

fig

# from these we see that: V1 -> green, V2 -> blue, V3 -> red



# plot inferred abundance distribution
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
    abund_l[:,1],
    abund_l[:,2],
    abund_l[:,3],
    color=[CairoMakie.RGBf(abund_l[i,3], abund_l[i, 1], abund_l[i,2]) for i ∈ 1:Npoints],
    marker=:circle,
    markersize = 10,
)

# the triangle is drawn from (0,0) to (0.5, sqrt(3)/2) to (1,0).
xlims!(ax, -0.2, 1.2) # to center the triangle and allow space for the labels
ylims!(ax, -0.3, 1.1)
text!(ax, Point2f(0.075, 0.5), text="z₃", fontsize=22)
text!(ax, Point2f(0.825, 0.5), text="z₂", fontsize=22)
text!(ax, Point2f(0.5, -0.175), text="z₁", fontsize=22)
hidedecorations!(ax) # to hide the axis decorations
fig

save(joinpath(figpath, "abundance-fit.png"), fig)
save(joinpath(figpath, "abundance-fit.pdf"), fig)




# ---------------------------------------------
# ------------- LINEAR NOISY MODEL-----------------
# ---------------------------------------------

figpath = joinpath(savepath, "linear-noisy")
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

text!(ax, Point2f(0.07, 0.5), text="Blue", fontsize=22)
text!(ax, Point2f(0.825, 0.5), text="Green", fontsize=22)
text!(ax, Point2f(0.45, -0.175), text="Red", fontsize=22)

save(joinpath(figpath, "abundance-orig.png"), fig)
save(joinpath(figpath, "abundance-orig.pdf"), fig)

fig

# visualize the data
fig = Figure();
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance")
for i ∈ 1:10:nrow(X)
    Rᵢ = Array(Xnoise[i,:])
    lines!(ax, λs, Rᵢ, color=CairoMakie.RGBAf(abund[:,i]..., 0.25), linewidth=1)
end
xlims!(ax, λs[1], λs[end])

fig

save(joinpath(figpath, "sample-spectra.png"), fig)
save(joinpath(figpath, "sample-spectra.pdf"), fig)


k = 75
m = 10
λ = 0.01
Nᵥ = 3

gsm_l = GSM(k=k, m=m, Nv=Nᵥ, λ=λ,  nonlinear=false, linear=true, bias=false, make_positive=true, tol=1e-9, nepochs=100, rand_init=true, rng=StableRNG(42))
mach_l = machine(gsm_l, Xnoise)
fit!(mach_l, verbosity=1)
abund_l = DataFrame(MLJ.transform(mach_l, Xnoise));


rpt = report(mach_l)
node_means = rpt[:node_data_means]
Q = rpt[:Q]
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]
stdev = sqrt(rpt[:β⁻¹])  # 0.010169833821451216 Nailed it!

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
ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");

lr = lines!(ax, λs, Rr, color=mints_colors[2], linewidth=3)
lg = lines!(ax, λs, Rg, color=mints_colors[1], linewidth=3)
lb = lines!(ax, λs, Rb, color=mints_colors[3], linewidth=3)

ls_fit = []
linestyles = [:solid, :dash, :dot]
i = 1
for idx ∈ idx_vertices
    Rout = node_means[:,idx]
    band!(ax, λs, Rout .- (2*stdev), Rout .+ (2*stdev), color=(:gray, 0.5))
    li = lines!(ax, λs, Rout, color=:gray, linewidth=2, linestyle=linestyles[i])
    push!(ls_fit, li)
    i += 1
end

fig[1,1] = Legend(fig, [lr, lg, lb, ls_fit...], ["Red Band", "Green Band", "Blue Band", "Vertex 1", "Vertex 2", "Vertex 3"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
xlims!(ax, λs[1], λs[end])
save(joinpath(figpath, "extracted-endmembers.png"), fig)
save(joinpath(figpath, "extracted-endmembers.pdf"), fig)

fig


# from these we see that: V1 -> red, V2 -> green, V3 -> blue


# plot distribution linear data
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
    abund_l[:,1],
    abund_l[:,2],
    abund_l[:,3],
    color=[CairoMakie.RGBf(abund_l[i,:]...) for i ∈ 1:Npoints],
    marker=:circle,
    markersize = 10,
)

# the triangle is drawn from (0,0) to (0.5, sqrt(3)/2) to (1,0).
xlims!(ax, -0.2, 1.2) # to center the triangle and allow space for the labels
ylims!(ax, -0.3, 1.1)
text!(ax, Point2f(0.075, 0.5), text="z₃", fontsize=22)
text!(ax, Point2f(0.825, 0.5), text="z₂", fontsize=22)
text!(ax, Point2f(0.5, -0.175), text="z₁", fontsize=22)
hidedecorations!(ax) # to hide the axis decorations
fig

save(joinpath(figpath, "abundance-fit.png"), fig)
save(joinpath(figpath, "abundance-fit.pdf"), fig)



# ---------------------------------------------
# -------- LINEAR ASYMMETRIC MODEL ------------
# ---------------------------------------------


figpath = joinpath(savepath, "linear-asymmetric")
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
    abund_2[1,:],
    abund_2[2,:],
    abund_2[3,:],
    color=[CairoMakie.RGBf(abund_2[:,i]...) for i ∈ 1:Npoints],
    marker=:circle,
    markersize = 8,
)

# the triangle is drawn from (0,0) to (0.5, sqrt(3)/2) to (1,0).
xlims!(ax, -0.2, 1.2) # to center the triangle and allow space for the labels
ylims!(ax, -0.3, 1.1)
hidedecorations!(ax) # to hide the axis decorations

text!(ax, Point2f(0.07, 0.5), text="Blue", fontsize=22)
text!(ax, Point2f(0.825, 0.5), text="Green", fontsize=22)
text!(ax, Point2f(0.45, -0.175), text="Red", fontsize=22)

save(joinpath(figpath, "abundance-orig.png"), fig)
save(joinpath(figpath, "abundance-orig.pdf"), fig)

fig


# visualize the data
fig = Figure();
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance")
for i ∈ 1:10:nrow(X)
    Rᵢ = Array(X2[i,:])
    lines!(ax, λs, Rᵢ, color=CairoMakie.RGBAf(abund_2[:,i]..., 0.25), linewidth=1)
end
xlims!(ax, λs[1], λs[end])

fig

save(joinpath(figpath, "sample-spectra.png"), fig)
save(joinpath(figpath, "sample-spectra.pdf"), fig)


k = 75
m = 10
# λ = 0.01
λ = 0.1
Nᵥ = 3

gsm_l = GSM(k=k, m=m, Nv=Nᵥ, λ=λ,  nonlinear=false, linear=true, bias=false, make_positive=true, tol=1e-9, nepochs=100, rand_init=true, rng=StableRNG(42))
mach_l = machine(gsm_l, X2)
fit!(mach_l, verbosity=1)
abund_l = DataFrame(MLJ.transform(mach_l, X2));


rpt = report(mach_l)
node_means = rpt[:node_data_means]
Q = rpt[:Q]
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]
stdev = sqrt(rpt[:β⁻¹])  # 0.010169833821451216 Nailed it!

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
ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");

lr = lines!(ax, λs, Rr, color=mints_colors[2], linewidth=3)
lg = lines!(ax, λs, Rg, color=mints_colors[1], linewidth=3)
lb = lines!(ax, λs, Rb, color=mints_colors[3], linewidth=3)

ls_fit = []
linestyles = [:solid, :dash, :dot]
i = 1
for idx ∈ idx_vertices
    Rout = node_means[:,idx]
    band!(ax, λs, Rout .- (2*stdev), Rout .+ (2*stdev), color=(:gray, 0.5))
    li = lines!(ax, λs, Rout, color=:gray, linewidth=2, linestyle=linestyles[i])
    push!(ls_fit, li)
    i += 1
end

fig[1,1] = Legend(fig, [lr, lg, lb, ls_fit...], ["Red Band", "Green Band", "Blue Band", "Vertex 1", "Vertex 2", "Vertex 3"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
xlims!(ax, λs[1], λs[end])

fig

save(joinpath(figpath, "extracted-endmembers.png"), fig)
save(joinpath(figpath, "extracted-endmembers.pdf"), fig)



# from these we see that: V1 -> red, V2 -> green, V3 -> blue


# plot distribution linear data
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
    abund_l[:,1],
    abund_l[:,2],
    abund_l[:,3],
    color=[CairoMakie.RGBf(abund_l[i,:]...) for i ∈ 1:Npoints],
    marker=:circle,
    markersize = 10,
)

# the triangle is drawn from (0,0) to (0.5, sqrt(3)/2) to (1,0).
xlims!(ax, -0.2, 1.2) # to center the triangle and allow space for the labels
ylims!(ax, -0.3, 1.1)
text!(ax, Point2f(0.075, 0.5), text="z₃", fontsize=22)
text!(ax, Point2f(0.825, 0.5), text="z₂", fontsize=22)
text!(ax, Point2f(0.5, -0.175), text="z₁", fontsize=22)
hidedecorations!(ax) # to hide the axis decorations
fig

save(joinpath(figpath, "abundance-fit.png"), fig)
save(joinpath(figpath, "abundance-fit.pdf"), fig)




# ---------------------------------------------
# ------------- NONLINEAR MODEL----------------
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

text!(ax, Point2f(0.07, 0.5), text="Blue", fontsize=22)
text!(ax, Point2f(0.825, 0.5), text="Green", fontsize=22)
text!(ax, Point2f(0.45, -0.175), text="Red", fontsize=22)

save(joinpath(figpath, "abundance-orig.png"), fig)
save(joinpath(figpath, "abundance-orig.pdf"), fig)

fig


# visualize the data
fig = Figure();
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance")
for i ∈ 1:10:nrow(X)
    Rᵢ = Array(Xnl[i,:])
    lines!(ax, λs, Rᵢ, color=CairoMakie.RGBAf(abund[:,i]..., 0.25), linewidth=1)
end
xlims!(ax, λs[1], λs[end])

fig

save(joinpath(figpath, "sample-spectra.png"), fig)
save(joinpath(figpath, "sample-spectra.pdf"), fig)



k = 75
m = 10
λ = 0.001
Nᵥ = 3

gsm_l = GSM(k=k, m=m, Nv=Nᵥ, λ=λ, nonlinear=true, linear=true, bias=false, make_positive=true, tol=1e-5, nepochs=100, rand_init=false, rng=StableRNG(42))
mach_l = machine(gsm_l, Xnl)
fit!(mach_l, verbosity=1)
abund_l = DataFrame(MLJ.transform(mach_l, Xnl));



rpt = report(mach_l)
node_means = rpt[:node_data_means]
Q = rpt[:Q]
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]
stdev = sqrt(rpt[:β⁻¹])  # 0.010159018676549504 Nailed it!


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
ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");

lr = lines!(ax, λs, Rr, color=mints_colors[2], linewidth=3)
lg = lines!(ax, λs, Rg, color=mints_colors[1], linewidth=3)
lb = lines!(ax, λs, Rb, color=mints_colors[3], linewidth=3)

ls_fit = []
linestyles = [:solid, :dash, :dot]
i = 1
for idx ∈ idx_vertices
    Rout = node_means[:,idx]
    band!(ax, λs, Rout .- (2*stdev), Rout .+ (2*stdev), color=(:gray, 0.5))
    li = lines!(ax, λs, Rout, color=:gray, linewidth=2, linestyle=linestyles[i])
    push!(ls_fit, li)
    i += 1
end

fig[1,1] = Legend(fig, [lr, lg, lb, ls_fit...], ["Red Band", "Green Band", "Blue Band", "Vertex 1", "Vertex 2", "Vertex 3"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
xlims!(ax, λs[1], λs[end])

fig

save(joinpath(figpath, "extracted-endmembers.png"), fig)
save(joinpath(figpath, "extracted-endmembers.pdf"), fig)


# plot distribution linear data
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
    abund_l[:,1],
    abund_l[:,2],
    abund_l[:,3],
    color=[CairoMakie.RGBf(abund_l[i,:]...) for i ∈ 1:Npoints],
    marker=:circle,
    markersize = 10,
)

# the triangle is drawn from (0,0) to (0.5, sqrt(3)/2) to (1,0).
xlims!(ax, -0.2, 1.2) # to center the triangle and allow space for the labels
ylims!(ax, -0.3, 1.1)
text!(ax, Point2f(0.075, 0.5), text="z₃", fontsize=22)
text!(ax, Point2f(0.825, 0.5), text="z₂", fontsize=22)
text!(ax, Point2f(0.5, -0.175), text="z₁", fontsize=22)
hidedecorations!(ax) # to hide the axis decorations
fig

save(joinpath(figpath, "abundance-fit.png"), fig)
save(joinpath(figpath, "abundance-fit.pdf"), fig)





# ---------------------------------------------
# ------------- LNEAR MODEL w/ GSMBig ---------
# ---------------------------------------------

figpath = joinpath(savepath, "linear-uniform")


k = 75
m = 10
λ = 0.001
Nᵥ = 3

n_nodes = binomial(k + Nᵥ - 2, Nᵥ - 1)
n_rbfs = binomial(m + Nᵥ - 2, Nᵥ - 1)

gsm_l = GSMBig(n_nodes=n_nodes, n_rbfs=n_rbfs, Nv=Nᵥ, λ=λ,  nonlinear=false, linear=true, bias=false, make_positive=true, tol=1e-9, nepochs=100, rand_init=false, rng=StableRNG(42))
mach_l = machine(gsm_l, X)

fit!(mach_l, verbosity=1)


abund_l = DataFrame(MLJ.transform(mach_l, X));
model = fitted_params(mach_l)[:gsm]

rpt = report(mach_l)
node_means = rpt[:node_data_means]
Q = rpt[:Q]
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]
stdev = sqrt(rpt[:β⁻¹])


model = fitted_params(mach_l)[:gsm]
πk = model.πk
Z = model.Z

# plot log likelihoods
fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Log-likelihood", yscale=log10)
lines!(ax, 2:length(llhs), llhs[2:end], linewidth=3)
fig

save(joinpath(figpath, "llh__BIG.png"), fig)
save(joinpath(figpath, "llh__BIG.pdf"), fig)

fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Q (a.u.)", yscale=log10)
lines!(ax, 2:length(Q), Q[2:end], linewidth=3)
fig

save(joinpath(figpath, "Q-fit__BIG.png"), fig)
save(joinpath(figpath, "Q-fit__BIG.pdf"), fig)


# plot endmembers for linear data
fig = Figure();
ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");

lr = lines!(ax, λs, Rr, color=mints_colors[2], linewidth=3)
lg = lines!(ax, λs, Rg, color=mints_colors[1], linewidth=3)
lb = lines!(ax, λs, Rb, color=mints_colors[3], linewidth=3)

ls_fit = []
linestyles = [:solid, :dash, :dot]
i = 1
for idx ∈ idx_vertices
    Rout = node_means[:,idx]
    band!(ax, λs, Rout .- (2*stdev), Rout .+ (2*stdev), color=(:gray, 0.5))
    li = lines!(ax, λs, Rout, color=:gray, linewidth=2, linestyle=linestyles[i])
    push!(ls_fit, li)
    i += 1
end

fig[1,1] = Legend(fig, [lr, lg, lb, ls_fit...], ["Red Band", "Green Band", "Blue Band", "Vertex 1", "Vertex 2", "Vertex 3"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
xlims!(ax, λs[1], λs[end])

save(joinpath(figpath, "extracted-endmembers__BIG.png"), fig)
save(joinpath(figpath, "extracted-endmembers__BIG.pdf"), fig)

fig

# from these we see that: V1 -> green, V2 -> blue, V3 -> red



# plot inferred abundance distribution
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
    abund_l[:,1],
    abund_l[:,2],
    abund_l[:,3],
    color=[CairoMakie.RGBf(abund_l[i,3], abund_l[i, 1], abund_l[i,2]) for i ∈ 1:Npoints],
    marker=:circle,
    markersize = 10,
)

# the triangle is drawn from (0,0) to (0.5, sqrt(3)/2) to (1,0).
xlims!(ax, -0.2, 1.2) # to center the triangle and allow space for the labels
ylims!(ax, -0.3, 1.1)
text!(ax, Point2f(0.075, 0.5), text="z₃", fontsize=22)
text!(ax, Point2f(0.825, 0.5), text="z₂", fontsize=22)
text!(ax, Point2f(0.5, -0.175), text="z₁", fontsize=22)
hidedecorations!(ax) # to hide the axis decorations
fig

save(joinpath(figpath, "abundance-fit__BIG.png"), fig)
save(joinpath(figpath, "abundance-fit__BIG.pdf"), fig)


