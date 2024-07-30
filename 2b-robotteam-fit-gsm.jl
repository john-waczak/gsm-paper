#using MLJ, GenerativeTopographicMapping
using MLJ, GenerativeTopographicMapping, MLJNonnegativeMatrixFactorization
using DataFrames, CSV
using StableRNGs
using Random
using JSON
using LinearAlgebra


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
outpath = "./output/robot-team/fits"
figpath = "./figures/robot-team"
mkpath(outpath)
mkpath(figpath)

include("./utils/fit-metrics.jl")

# LOAD IN  DATASET
readdir("./data/robot-team/")
df_features = CSV.read("./data/robot-team/df_features.csv", DataFrame);
nrow(df_features)


# Explore data and infer intrinsic dimensionality (i.e. # of endmembers)
println("\tAssessing dimensionality...")

Svd = svd(Array(df_features) .- mean(Array(df_features), dims=1));
σs = abs2.(Svd.S) ./ (nrow(df_features) - 1)
variances =  σs ./ sum(σs)
cum_var = cumsum(σs ./ sum(σs))

fig = Figure();
ax = Axis(fig[2,1], xlabel="Component", ylabel="Explained Variance (%)", xticks=(1:length(σs)), xminorgridvisible=false, yminorgridvisible=false);
barplot!(ax, 1:length(σs), variances .* 100, color=mints_colors[1], strokecolor=:darkgreen, strokewidth=2)
hl = hlines!(ax, [1,], color=mints_colors[2], linestyle=:solid, linewidth=2, label="1%")
fig[1,1] = Legend(fig, [hl,], ["1 %",], framevisible=false, orientation=:horizontal, padding=(0,0,-20,0), labelsize=14, height=-5, halign=:right)
xlims!(ax, 0.5, 10)
ylims!(ax, -0.01, nothing)
fig

save(joinpath(figpath, "pca-variance.png"), fig)


# Based on this, between 4-8 endmembers should be fine
n_nodes = 3000
n_rbfs = 250

#k = 25 # this roughly corresponds to 3000 nodes
Nv_s = [3, 4, 5, 6]
s = 1/10
λe_s = [1.0, 0.1, 0.01, 0.001,]
λw_s = [1_000., 100, 10, 1, 0.1,]

Yorig = Matrix(df_features);

println("\tNumber of models to train: ", length(Nv_s) * length(λe_s) * length(λw_s))
println("\tTraining models...")

for Nv ∈ Nv_s
    for λe ∈ λe_s
        for λw ∈ λw_s
            json_path = joinpath(outpath, "fit_Nv-$(Nv)_λe-$(λe)_λw-$(λw).json")
            jlso_path = joinpath(outpath, "model_Nv-$(Nv)_λe-$(λe)_λw-$(λw).jlso")

            # only fit it if we haven't already
            if !isfile(json_path)
                println("\n")
                println("Nv: ", Nv, "\tλe: ", λe, "\tλw: ", λw)
                println("\n")

                gsm = GSMBigCombo(n_nodes=n_nodes, n_rbfs=n_rbfs, λe=λe, λw=λw, s=s, Nv=Nv, niters=10, tol=1e-6, nepochs=500, rng=StableRNG(42))

                mach = machine(gsm, df_features)
                fit!(mach, verbosity=1)

                # generate report
                rpt = report(mach)

                # collect results
                res = Dict()
                res[:Q] = rpt[:Q]
                res[:llhs] = rpt[:llhs]
                res[:converged] = rpt[:converged]
                res[:AIC] = rpt[:AIC]
                res[:BIC] = rpt[:BIC]
                res[:Nv] = Nv
                res[:s] = s
                res[:λe] = λe
                res[:λw] = λw
                Ŷ = data_reconstruction(mach, df_features)
                res[:rmse_reconst] = rmse(Ŷ, Yorig)

                # save the report for later comparison
                open(json_path, "w") do f
                    JSON.print(f, res)
                end
            end
        end
    end
end


# Load in results and create single DataFrame
df_res = []
for (root, dirs, files) ∈ walkdir(outpath)
    for file ∈ files
        if endswith(file, ".json")
            res = JSON.parsefile(joinpath(root, file))

            # generate path to jlso...
            fsave = replace(file, "fit" => "model", ".json" => ".jlso")
            fsave = joinpath(root, fsave)

            res["Q"] = res["Q"][end]
            res["llhs"] = res["llhs"][end]

            res["res_path"] = joinpath(root, file)
            res["mdl_path"] = joinpath(root, fsave)

            push!(df_res, res)
        end
    end
end


df_res = DataFrame(df_res)
CSV.write(joinpath("./output/robot-team/", "df_results.csv"), df_res)

unique(df_res.λw)

sort(df_res[df_res.λw .== 10.0 , [:Nv, :λe, :λw, :BIC, :AIC, :Q]], :BIC)


df_best = sort(df_res[df_res.λw .≥ 1.0 , :], :BIC);

df_out = df_best[1:10, [:Nv, :λe, :λw, :BIC, :AIC, :rmse_reconst,]]
CSV.write("./paper/fit-res.csv", df_out)
# df_best = df_res
# Identify the best Nv via BIC, AIC, Q, LLH

df_best[argmax(df_best.Q), [:Q, :Nv, :λe, :λw]]

df_best[argmax(df_best.llhs),  [:llhs, :Nv, :λe, :λw]]
df_best[argmin(df_best.AIC), [:AIC, :Nv, :λe, :λw]]





# Based on this, between 4-8 endmembers should be fine
outpath = "./output/robot-team/fits-linear"
mkpath(outpath)
k = 75
n_nodes = binomial(k+3-2, 3-1)
Nv_s = [3, 4, 5, 6, 7, 8]
λ_s = [1.0, 0.1, 0.01, 0.001,]

Yorig = Matrix(df_features);
println("\tNumber of models to train: ", length(Nv_s) * length(λ_s))
println("\tTraining models...")

for Nv ∈ Nv_s
    for λ ∈ λ_s
        json_path = joinpath(outpath, "fit_Nv-$(Nv)_λ-$(λ).json")
        jlso_path = joinpath(outpath, "model_Nv-$(Nv)_λ-$(λ).jlso")

        # only fit it if we haven't already
        if !isfile(json_path)
            println("\n")
            println("Nv: ", Nv, "\tλ: ", λ)
            println("\n")

            gsm = GSMBigLinear(n_nodes=n_nodes, λ=λ, Nv=Nv, niters=10, tol=1e-6, nepochs=500, rng=StableRNG(42))
            mach = machine(gsm, df_features)
            fit!(mach, verbosity=1)

            # generate report
            rpt = report(mach)

            # collect results
            res = Dict()
            res[:Q] = rpt[:Q]
            res[:llhs] = rpt[:llhs]
            res[:converged] = rpt[:converged]
            res[:AIC] = rpt[:AIC]
            res[:BIC] = rpt[:BIC]
            res[:Nv] = Nv
            res[:λ] = λ
            Ŷ = data_reconstruction(mach, df_features)
            res[:rmse_reconst] = rmse(Ŷ, Yorig)

            # save the report for later comparison
            open(json_path, "w") do f
                JSON.print(f, res)
            end
        end
    end
end


# Load in results and create single DataFrame
df_res = []
for (root, dirs, files) ∈ walkdir(outpath)
    for file ∈ files
        if endswith(file, ".json")
            res = JSON.parsefile(joinpath(root, file))

            # generate path to jlso...
            fsave = replace(file, "fit" => "model", ".json" => ".jlso")
            fsave = joinpath(root, fsave)

            res["Q"] = res["Q"][end]
            res["llhs"] = res["llhs"][end]

            res["res_path"] = joinpath(root, file)
            res["mdl_path"] = joinpath(root, fsave)

            push!(df_res, res)
        end
    end
end


df_res = DataFrame(df_res)
CSV.write(joinpath("./output/robot-team/", "df_results.csv"), df_res)


df_best = df_res[df_res.λw .≥ 1.0 , :]
df_best = df_res
# Identify the best Nv via BIC, AIC, Q, LLH

df_best[argmax(df_best.Q), [:Q, :Nv, :λe, :λw]]
df_best[argmax(df_best.llhs),  [:llhs, :Nv, :λe, :λw]]
df_best[argmin(df_best.BIC), [:BIC, :Nv, :λe, :λw]]
df_best[argmin(df_best.AIC), [:AIC, :Nv, :λe, :λw]]





# Now lets do a deep dive for just Nv=3
# outpath = "./output/robot-team/fits-Nv=4"
# mkpath(outpath)

# Nv = 4
# k = 75
# m_s = [5, 10, 15, 20, 25,]
# λe = [0.1, 0.01, 0.001]
# λw_s = [100.0, 10.0, 1.0]


# Nv = 4
# s_s = [0.5, 1, 2, 3,] ./ 10
# n_nodes = 3000
# n_rbfs = 250
# λe_s = [0.01,]
# λw_s = [100.0, 10.0, 1.0]

outpath = "./output/robot-team/fits-Nv=3"
mkpath(outpath)

Nv = 3
k = 25
m_s = [5,10,15]
λe_s = [1.0, 0.1, 0.01, 0.001,]
λw_s = [1_000.0, 1000.0, 10.0, 1.0, 0.1]

length(m_s)*length(λe_s)*length(λw_s)
Yorig = Matrix(df_features);

for m ∈ m_s
    for λw ∈ λw_s
        for λe ∈ λe_s
            json_path = joinpath(outpath, "fit_Nv-$(Nv)_m-$(m)_λe-$(λe)_λw-$(λw).json")
            save_path = joinpath(outpath, "model_Nv-$(Nv)_m-$(m)_λe-$(λe)_λw-$(λw).jls")

            # only fit it if we haven't already
            if !isfile(json_path)
                println("m: ", m, "\tλe: ", λe, "\tλw: ", λw)

                # gsm = GSMBigCombo(n_nodes=n_nodes, n_rbfs=n_rbfs, s=s, Nv=Nv, λe=λe, λw=λw, niters=10, tol=1e-6, nepochs=500, rng=StableRNG(42))
                gsm = GSMCombo(k=k, m=m, Nv=Nv, λe=λe, λw=λw, niters=10, tol=1e-6, nepochs=1000, rng=StableRNG(42))
                mach = machine(gsm, df_features)
                println("\ttraining...")
                fit!(mach, verbosity=0)
                println("\t...finished.")

                # generate report
                rpt = report(mach)

                # collect results
                res = Dict()
                res[:Q] = rpt[:Q]
                res[:llhs] = rpt[:llhs]
                res[:converged] = rpt[:converged]
                res[:AIC] = rpt[:AIC]
                res[:BIC] = rpt[:BIC]
                res[:Nv] = Nv
                res[:λe] = λe
                res[:λw] = λw
                res[:m] = m

                W = rpt[:W]
                res[:W_nonlinear_mean] = mean(W[:, Nv+1:end])
                res[:W_nonlinear_median] = median(W[:, Nv+1:end])
                res[:W_nonlinear_min] = minimum(W[:, Nv+1:end])
                res[:W_nonlinear_max] = maximum(W[:, Nv+1:end])

                Ŷ = data_reconstruction(mach, df_features)
                res[:rmse_reconst] = rmse(Ŷ, Yorig)

                println("\tReconstruction rmse: ", res[:rmse_reconst], "\tBIC: ", res[:BIC])

                # save the report for later comparison
                open(json_path, "w") do f
                    JSON.print(f, res)
                end

                # Save the machine for later use
                # https://juliaai.github.io/MLJ.jl/stable/machines/#Saving-machines
                println("\t\tsaving...")
                MLJ.save(save_path, mach)
                println("\t\t...complete")
            end
        end
    end
end



# Load in results and create single DataFrame
df_res = []
for (root, dirs, files) ∈ walkdir(outpath)
    for file ∈ files
        if endswith(file, ".json") .&& !occursin("final", file)
            res = JSON.parsefile(joinpath(root, file))

            # generate path to jlso...
            fsave = replace(file, "fit" => "model", ".json" => ".jlso")
            fsave = joinpath(root, fsave)

            res["Q"] = res["Q"][end]
            res["llhs"] = res["llhs"][end]
            res["res_path"] = joinpath(root, file)
            res["mdl_path"] = joinpath(root, fsave)

            push!(df_res, res)
        end
    end
end

df_res = DataFrame(df_res)

CSV.write(joinpath("./output/robot-team/", "df_results-Nv=$(Nv).csv"), df_res)


# Identify the best Nv via BIC, AIC, Q, LLH
df_best = df_res[df_res.λw .≥ 1.0, :]
df_best[argmax(df_best.Q), [:Q, :Nv, :λe, :λw, :m]]
df_best[argmax(df_best.llhs),  [:llhs, :Nv, :λe, :λw, :m]]
df_best[argmin(df_best.BIC), [:BIC, :Nv, :λe, :λw, :m]]
df_best[argmin(df_best.AIC), [:AIC, :Nv, :λe, :λw, :m]]
df_best[argmin(df_best.rmse_reconst), [:rmse_reconst, :Nv, :λe, :λw, :m]]

sort(df_res[df_res.λw .>= 1.0, [:λw, :λe, :BIC, :Q, :rmse_reconst, :W_nonlinear_median, :W_nonlinear_max, :m]], :BIC)


# train final model:
outpath = "./output/robot-team"
mkpath(outpath)

Nv = 3
k = 75
m = 10
λe = 0.01
λw = 1.0

json_path = joinpath(outpath, "fit_final.json")
save_path = joinpath(outpath, "model_final.jls")

gsm = GSMCombo(k=k, m=m, Nv=Nv, λe=λe, λw=λw, niters=10, tol=1e-6, nepochs=3_000, rng=StableRNG(42))

mach = machine(gsm, df_features)
fit!(mach, verbosity=1)

# generate report
rpt = report(mach)

# collect results
res = Dict()
res[:Q] = Float64.(rpt[:Q])
res[:llhs] = Float64.(rpt[:llhs])
res[:converged] = rpt[:converged]
res[:AIC] = rpt[:AIC]
res[:BIC] = rpt[:BIC]
res[:Nv] = Nv
res[:λe] = λe
res[:λw] = λw
res[:s] = s
res[:n_rbfs] = n_rbfs
res[:n_nodes] = n_nodes
res[:β⁻¹] = rpt[:β⁻¹]

W = rpt[:W]
res[:W_nonlinear_mean] = mean(W[:, Nv+1:end])
res[:W_nonlinear_median] = median(W[:, Nv+1:end])
res[:W_nonlinear_min] = minimum(W[:, Nv+1:end])
res[:W_nonlinear_max] = maximum(W[:, Nv+1:end])
Ŷ = data_reconstruction(mach, df_features)
res[:rmse_reconst] = rmse(Ŷ, Yorig)


# save the report for later comparison
open(json_path, "w") do f
    JSON.print(f, res)
end

# Save the machine for later use
# https://juliaai.github.io/MLJ.jl/stable/machines/#Saving-machines
println("\t\tsaving...")
MLJ.save(save_path, mach)
println("\t\t...complete")



# plot log likelihoods
fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Log-likelihood", yscale=log10)
lines!(ax, 2:length(res[:llhs]), Float64.(res[:llhs][2:end]), linewidth=3)
# xlims!(ax, 0, 200)
fig

save(joinpath(figpath, "llh.png"), fig)

fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Q (a.u.)", yscale=log10)
lines!(ax, 2:length(res[:Q]), Float64.(res[:Q][2:end]), linewidth=3)
# xlims!(0, 200)
fig

save(joinpath(figpath, "Q-fit.png"), fig)


# let's compute the mean values of the weights to see if they are sparse
W = rpt[:W]

extrema(W[:, Nv+1:end])
mean(W[:, Nv+1:end])
median(W[:, Nv+1:end])

Ypred = data_reconstruction(mach, df_features)
Zpred = DataFrame(MLJ.transform(mach, df_features))

idx1_max = argmax(Zpred.Z1)
idx2_max = argmax(Zpred.Z2)
idx3_max = argmax(Zpred.Z3)

Zpred[[idx1_max, idx2_max, idx3_max], :]

idx_vertices = rpt[:idx_vertices]
node_means = rpt[:node_data_means]

fig = Figure();
ax = Axis(fig[1,1], xlabel="λ (nm)", ylabel="Reflectance")

for i ∈ 1:25:size(Yorig, 1)
    lines!(ax, wavelengths, Yorig[i,:], linewidth=1, color=:gray, alpha=0.3)
end

lines!(ax, wavelengths, node_means[:,idx_vertices[1]], color=:red)
lines!(ax, wavelengths, node_means[:,idx_vertices[2]], color=:green)
lines!(ax, wavelengths, node_means[:,idx_vertices[3]], color=:blue)

lines!(ax, wavelengths, Ypred[idx1_max,:], color=:red, linestyle=:dash)
lines!(ax, wavelengths, Ypred[idx2_max,:], color=:green, linestyle=:dash)
lines!(ax, wavelengths, Ypred[idx3_max,:], color=:blue, linestyle=:dash)

fig



# NOTE: We need to do this using idx_vertices
# I think this means that we are not updating W
# correctly since we really need to be


idx_vertices = rpt[:idx_vertices]
node_means = rpt[:node_data_means]
node_means_2 = (rpt[:Φ] * rpt[:W]')
all(node_means[:, idx_vertices[1]] .== node_means_2[idx_vertices[1], :])
W = rpt[:W]
Φ = rpt[:Φ]
node_means_2 = Φ[idx_vertices, :] * W'
findall(Φ[idx_vertices, :].== 0)
findall(Φ[idx_vertices, 4:end] .!= 0.0)
all((W*Φ') .== (Φ*W')')
all(rpt[:node_data_means] .==  rpt[:W]*rpt[:Φ]')


# nmf = NMF(k=Nv, cost=:Euclidean, normalize_abundance=true, tol=1e-6, maxiters=1000, rng=StableRNG(42))
# mach = machine(nmf, df_features)
# fit!(mach, verbosity=1)

# fp = fitted_params(mach)
H = fp.H
W = fp.W

fig = Figure();
ax = Axis(fig[1,1], xlabel="λ (nm)", ylabel="Reflectance")

for i ∈ 1:25:size(Yorig, 1)
    lines!(ax, wavelengths, Yorig[i,:], linewidth=1, color=:gray, alpha=0.3)
end

lines!(ax, wavelengths, W[:,1]) # ./ maximum(W[:,1]))
lines!(ax, wavelengths, W[:,2]) # ./ maximum(W[:,2]))
lines!(ax, wavelengths, W[:,3]) # ./ maximum(W[:,3]))

fig


