include("../src/GenerativeTopographicMapping.jl")

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







k = 50
m = 10
λ = 0.001
Nᵥ = 3

# visualize the prior distribution
αs = [0.1, 0.5, 0.75, 1.0, 5.0]
for α ∈ αs
    α = α * ones(Nᵥ)

    Z = GenerativeTopographicMapping.get_barycentric_grid_coords(k, Nᵥ)
    πk = zeros(size(Z, 2))
    f_dirichlet = Dirichlet(α)

    e = 0.5 * (1/k)  # offset to deal with Inf value on boundary

    for j ∈ axes(Z,2)
        p = Z[:,j]
        for i ∈ axes(Z,1)
            if Z[i,j] == 0.0
                p[i] = e
            end
        end
        p = p ./ sum(p)
        πk[j] = pdf(f_dirichlet, p)
    end

    πk = πk ./ sum(πk)

    fig = Figure();
    ax = Axis(fig[1, 1], aspect=1, title="p(zₖ) with α=$(α)", titlefont=:regular, titlealign=:left);

    ternaryaxis!(
        ax,
        tick_fontsize=15,
    );

    ts = ternaryscatter!(
        ax,
        Z[1,:],
        Z[2,:],
        Z[3,:],
        color = πk,
        marker = :circle,
        markersize = 9,
    )

    xlims!(ax, -0.2, 1.2)
    ylims!(ax, -0.3, 1.1)
    hidedecorations!(ax)
    hidespines!(ax)
    text!(ax, Point2f(0.1, 0.5), text="z₃", fontsize=22)
    text!(ax, Point2f(0.825, 0.5), text="z₂", fontsize=22)
    text!(ax, Point2f(0.5, -0.175), text="z₁", fontsize=22)

    cb = Colorbar(fig[1,2], ts, label="πₖ")

    fig

    save("./figures/prior_$(α[1]).png", fig)
    save("./figures/prior_$(α[1]).pdf", fig)

end








# Test 1: Linear Mixing
λs = range(350, stop=750, length=1000)
# λs = range(400, stop=700, length=1000)

Rb = exp.(-(λs .- 460.0).^2 ./(2*(22)^2))
Rg = exp.(-(λs .- 525.0).^2 ./(2*(28)^2))
Rr = exp.(-(λs .- 625.0).^2 ./(2*(20)^2))

# Rb = exp.(-(abs.(λs .- 450.0)./ 10).^1)
# Rg = exp.(-(abs.(λs .- 530.0)./ 30).^2)
# Rr = exp.(-(abs.(λs .- 620.0)./ 40).^10)

# Rb = exp.(-(abs.(λs .- 476.0)./ 10).^1)
# Rg = exp.(-(abs.(λs .- 530.0)./ 30).^2)
# Rr = exp.(-(abs.(λs .- 605.0)./ 40).^10)

Rb .= Rb./maximum(Rb)
Rg .= Rg./maximum(Rg)
Rr .= Rr./maximum(Rr)

fig = Figure();
ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");
lr = lines!(ax, λs, Rr, color=mints_colors[2], linewidth=2)
lg = lines!(ax, λs, Rg, color=mints_colors[1], linewidth=2)
lb = lines!(ax, λs, Rb, color=mints_colors[3], linewidth=2)
fig[1,1] = Legend(fig, [lr, lg, lb], ["Red Band", "Green Band", "Blue Band"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
fig

save("./figures/rbg-orig.png", fig)
save("./figures/rbg-orig.pdf", fig)



# generate dataframe of reflectance values from combined dataset

# Npoints = 10_000
Npoints = 25_000
α_true = 0.05 * ones(3)
f_dir = Dirichlet(α_true)
abund = rand(rng, f_dir, Npoints)



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
    markersize = 10,
)

# the triangle is drawn from (0,0) to (0.5, sqrt(3)/2) to (1,0).
xlims!(ax, -0.2, 1.2) # to center the triangle and allow space for the labels
ylims!(ax, -0.3, 1.1)
hidedecorations!(ax) # to hide the axis decorations

text!(ax, Point2f(0.075, 0.5), text="Blue", fontsize=22)
text!(ax, Point2f(0.825, 0.5), text="Green", fontsize=22)
text!(ax, Point2f(0.5, -0.175), text="Red", fontsize=22)

fig

save("./figures/abundance-orig.png", fig)
save("./figures/abundance-orig.pdf", fig)

X = zeros(Npoints, length(λs));    # linear mixing
Xnoise = zeros(Npoints, length(λs));
X2 = zeros(Npoints, length(λs));   # non-linear mixing
for i ∈ axes(X, 1)
    X[i,:] .= (abund[1,i] .* Rr) .+ (abund[2,i] .* Rg) .+ (abund[3,i] .* Rb)
    # Xnoise[i,:] .= X[i,:] .+ 0.01*(2 .* (rand(rng, length(λs)) .- 0.5))
    Xnoise[i,:] .= X[i,:] .+ rand(rng, Normal(0, 0.01), length(λs))

    # NOTE: these are random values ∈ ± 0.01
    #       therefore the std of the values
    #       should be (b-a)²/12 ≈ 0.005773502691896258

    X2[i,:] .= (abund[1,i] .* Rr) .+ (abund[2,i] .* Rg) .+ (abund[3,i] .* Rb) .+ (abund[1,i] * abund[2,i] .* Rr .* Rg)  .+ (abund[1,i] * abund[2,i] .* Rr .* Rg) .+ (abund[1,i] * abund[3,i] .* Rr .* Rb)  .+ (abund[2,i] * abund[3,i] .* Rg .* Rb)

    # X[i,:] .= X[i,:] ./ maximum(X[i,:])
    # X2[i,:] .= X2[i,:] ./ maximum(X2[i,:])
end

names = Symbol.(["λ_" * lpad(i, 3, "0") for i ∈ 1:length(λs)])
X = DataFrame(X, names);
Xnoise = DataFrame(Xnoise, names);
X2 = DataFrame(X2, names);

fig = Figure();
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance")
for i ∈ 1:10:nrow(X)
    Rᵢ = Array(X[i,:])
    lines!(ax, λs, Rᵢ, color=CairoMakie.RGBAf(abund[:,i]..., 0.25), linewidth=1)
end
fig

save("./figures/sample-spectra.png", fig)
save("./figures/sample-spectra.pdf", fig)


fig = Figure();
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance")
for i ∈ 1:10:nrow(X2)
    Rᵢ = Array(X2[i,:])
    lines!(ax, λs, Rᵢ, color=CairoMakie.RGBAf(abund[:,i]..., 0.25), linewidth=1)
end
fig

save("./figures/sample-spectra-nonlinear.png", fig)
save("./figures/sample-spectra-nonlinear.pdf", fig)



# ---------------------------------------------
# ------------- LINEAR MODEL -----------------
# ---------------------------------------------

if !ispath("./figures/linear")
    mkpath("./figures/linear")
end

k = 10
m = 5
λ = 0.001
Nᵥ = 5

# Do inital fit with Nᵥ > Nᵥ_true to test dimensionality identification
gsm_l = GenerativeTopographicMapping.GSM(k=k, m=m, Nv=Nᵥ, λ=λ,  nonlinear=false, linear=true, bias=false, make_positive=true, tol=1e-5, nepochs=25, rand_init=false, rng=StableRNG(42))
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

save("./figures/linear/extra-endmembers-boxplot.png", fig)
save("./figures/linear/extra-endmembers-boxplot.pdf", fig)


k = 75
m = 10
λ = 0.001
Nᵥ = 3

gsm_l = GenerativeTopographicMapping.GSM(k=k, m=m, Nv=Nᵥ, λ=λ,  nonlinear=false, linear=true, bias=false, make_positive=true, tol=1e-9, nepochs=100, rand_init=false, rng=StableRNG(42))
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

save("./figures/linear/lllh-linear.png",fig)
save("./figures/linear/lllh-linear.pdf",fig)


fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Q (a.u.)", yscale=log10)
lines!(ax, 2:length(Q), Q[2:end], linewidth=3)
fig

save("./figures/linear/Q-fit.png",fig)
save("./figures/linear/Q-fit.pdf",fig)



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
hidedecorations!(ax) # to hide the axis decorations
fig

save("./figures/linear/abundance-fit.png", fig)
save("./figures/linear/abundance-fit.pdf", fig)


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
    # Rout .= Rout ./ maximum(Rout)
    band!(ax, λs, Rout .- (2*stdev), Rout .+ (2*stdev), color=(:gray, 0.5))
    li = lines!(ax, λs, Rout, color=:gray, linewidth=2, linestyle=linestyles[i])
    push!(ls_fit, li)
    i += 1
end

fig[1,1] = Legend(fig, [lr, lg, lb, ls_fit...], ["Red Band", "Green Band", "Blue Band", "Vertex 1", "Vertex 2", "Vertex 3"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)

fig


save("./figures/linear/endmembers-extracted.png", fig)
save("./figures/linear/endmembers-extracted.pdf", fig)



# ---------------------------------------------
# ------------- LINEAR NOISY MODEL-----------------
# ---------------------------------------------

if !ispath("./figures/linear-noisy")
    mkpath("./figures/linear-noisy")
end



k = 75
m = 10
λ = 0.01
Nᵥ = 3

gsm_l = GenerativeTopographicMapping.GSM(k=k, m=m, Nv=Nᵥ, λ=λ,  nonlinear=false, linear=true, bias=false, make_positive=true, tol=1e-9, nepochs=100, rand_init=true, rng=StableRNG(42))
mach_l = machine(gsm_l, Xnoise)
fit!(mach_l, verbosity=1)
abund_l = DataFrame(MLJ.transform(mach_l, Xnoise));


rpt = report(mach_l)
node_means = rpt[:node_data_means]
Q = rpt[:Q]
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]
stdev = sqrt(rpt[:β⁻¹])  # 0.010159018676549504 Nailed it!

# λ = 0.01
# iter: 46, log-likelihood = 6.49735207780663e11
# sqrt(1/β) = 0.010412048012862687

# λ = 0.001
# iter: 46, log-likelihood = 6.49735207780663e11
# sqrt(1/β) = 0.010412048012862687

# λ = 0.0001
# iter: 46, log-likelihood = 6.497352077878674e11
# sqrt(1/β) = 0.010412048012885446


# plot log likelihoods
fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Log-likelihood", yscale=log10)
lines!(ax, 3:length(llhs), llhs[3:end], linewidth=3)
fig
save("./figures/linear-noisy/lllh-linear.png",fig)
save("./figures/linear-noisy/lllh-linear.pdf",fig)


fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Q (a.u.)", yscale=log10)
lines!(ax, 2:length(Q), Q[2:end], linewidth=3)
fig

save("./figures/linear-noisy/Q-fit.png",fig)
save("./figures/linear-noisy/Q-fit.pdf",fig)



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
hidedecorations!(ax) # to hide the axis decorations
fig

save("./figures/linear-noisy/abundance-fit.png", fig)
save("./figures/linear-noisy/abundance-fit.pdf", fig)


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
    # Rout .= Rout ./ maximum(Rout)
    band!(ax, λs, Rout .- (2*stdev), Rout .+ (2*stdev), color=(:gray, 0.5))
    li = lines!(ax, λs, Rout, color=:gray, linewidth=2, linestyle=linestyles[i])
    push!(ls_fit, li)
    i += 1
end

fig[1,1] = Legend(fig, [lr, lg, lb, ls_fit...], ["Red Band", "Green Band", "Blue Band", "Vertex 1", "Vertex 2", "Vertex 3"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)

fig

save("./figures/linear-noisy/endmembers-extracted.png", fig)
save("./figures/linear-noisy/endmembers-extracted.pdf", fig)




# ---------------------------------------------
# ------------- NONLINEAR MODEL----------------
# ---------------------------------------------

if !ispath("./figures/nonlinear")
    mkpath("./figures/nonlinear")
end



k = 75
m = 10
s = 1
λe = 0.001
λw = 0.5
Nᵥ = 3

# gsm_l = GenerativeTopographicMapping.GSMNonlinear(k=k, m=m, Nv=Nᵥ, λe=λe, λw=λw, make_positive=true, tol=1e-5, nepochs=50, rng=StableRNG(42))
gsm_l = GenerativeTopographicMapping.GSM(k=k, m=m, Nv=Nᵥ, λ=0.001, nonlinear=true, linear=true, bias=false, make_positive=true, tol=1e-5, nepochs=50, rng=StableRNG(42))
mach_l = machine(gsm_l, X2)
fit!(mach_l, verbosity=1)
abund_l = DataFrame(MLJ.transform(mach_l, X2));


rpt = report(mach_l)
node_means = rpt[:node_data_means]
Q = rpt[:Q]
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]
stdev = sqrt(rpt[:β⁻¹])  # 0.010159018676549504 Nailed it!


# plot log likelihoods
fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Log-likelihood", yscale=log10)
lines!(ax, 3:length(llhs), llhs[3:end], linewidth=3)
fig
save("./figures/nonlinear/lllh-fit.png",fig)
save("./figures/nonlinear/lllh-fit.pdf",fig)


fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Q (a.u.)", yscale=log10)
lines!(ax, 2:length(Q), Q[2:end], linewidth=3)
fig

save("./figures/nonlinear/Q-fit.png",fig)
save("./figures/nonlinear/Q-fit.pdf",fig)



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
hidedecorations!(ax) # to hide the axis decorations
fig

save("./figures/nonlinear/abundance-fit.png", fig)
save("./figures/nonlinear/abundance-fit.pdf", fig)


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
    # Rout .= Rout ./ maximum(Rout)
    band!(ax, λs, Rout .- (2*stdev), Rout .+ (2*stdev), color=(:gray, 0.5))
    li = lines!(ax, λs, Rout, color=:gray, linewidth=2, linestyle=linestyles[i])
    push!(ls_fit, li)
    i += 1
end

fig[1,1] = Legend(fig, [lr, lg, lb, ls_fit...], ["Red Band", "Green Band", "Blue Band", "Vertex 1", "Vertex 2", "Vertex 3"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)

fig

save("./figures/nonlinear/endmembers-extracted.png", fig)
save("./figures/nonlinear/endmembers-extracted.pdf", fig)


