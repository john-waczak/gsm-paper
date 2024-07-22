using MLJ, GenerativeTopographicMapping
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
hl = hlines!(ax, [1,], color=mints_colors[2], linestyle=:solid, linewidth=2, label="1%")
barplot!(ax, 1:length(σs), variances .* 100, color=mints_colors[1], strokecolor=:darkgreen, strokewidth=2)
fig[1,1] = Legend(fig, [hl,], ["1 %",], framevisible=false, orientation=:horizontal, padding=(0,0,-20,0), labelsize=14, height=-5, halign=:right)
xlims!(ax, 0.5, 10)
ylims!(ax, -0.01, nothing)
fig

save(joinpath(figpath, "pca-variance.png"), fig)


fig
# Based on this, between 4-8 endmembers should be fine
n_nodes = 3000
n_rbfs = 250


#k = 25 # this roughly corresponds to 3000 nodes
# Nv_s = [2, 3, 4, 5, 6, 7, 8]
Nv_s = [3, 4, 5, 6, 7, 8]
s = 1/10
# λe_s = [0.1, 0.01, 0.001, 0.0001, 0.00001]
# λw_s = [10_000, 1_000, 100, 10, 1, 0.1, 0.01, 0.001]
λe_s = [0.1, 0.01, 0.001,]
λw_s = [100, 10, 1, 0.1,]


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

                gsm = GSMBigCombo(n_nodes=n_nodes, n_rbfs=n_rbfs, λe=λe, λw=λw, s=s, Nv=Nv, niters=10, tol=1e-9, nepochs=100)

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




unique(df_res.λe)
unique(df_res.λw)
names(df_res)

# Identify the best Nv via BIC, AIC, Q, LLH

df_res[argmax(df_res.Q), [:Q, :Nv, :λe, :λw]]
df_res[argmax(df_res.llhs),  [:llhs, :Nv, :λe, :λw]]
df_res[argmin(df_res.BIC), [:BIC, :Nv, :λe, :λw]]
df_res[argmin(df_res.AIC), [:AIC, :Nv, :λe, :λw]]


df_res_10000 = df_res[(df_res.λw .== 10000), :]
df_res_10000[argmin(df_res_10000.Q), [:Q, :Nv, :λe]]
df_res_10000[argmin(df_res_10000.BIC), [:BIC, :Nv, :λe]]

df_res_1000 = df_res[(df_res.λw .== 1000), :]
df_res_1000[argmin(df_res_1000.Q), [:Q, :Nv, :λe]]
df_res_1000[argmin(df_res_1000.BIC), [:BIC, :Nv, :λe]]

df_res_100 = df_res[(df_res.λw .== 100), :]
df_res_100[argmin(df_res_100.Q), [:Q, :Nv, :λe]]
df_res_100[argmin(df_res_100.BIC), [:BIC, :Nv, :λe]]





# Now lets do a deep dive for just Nv=3
outpath = "./output/robot-team/fits-Nv=3"
mkpath(outpath)

Nv = 3
k = 75
m = 25
λe_s = [0.1, 0.01, 0.001]
λw_s = [10_000.0, 1000.0, 100., 10., 1., 0.1, 0.01, 0.001, 0.0001]

Yorig = Matrix(df_features);

for λe ∈ λe_s
    for λw ∈ λw_s
        json_path = joinpath(outpath, "fit_Nv-$(Nv)_λe-$(λe)_λw-$(λw).json")
        save_path = joinpath(outpath, "model_Nv-$(Nv)_λe-$(λe)_λw-$(λw).jls")

        # only fit it if we haven't already
        if !isfile(json_path)
            println("Nv: ", Nv, "\tλe: ", λe, "\tλw: ", λw)

            gsm = GSMCombo(k=k, m=m, Nv=Nv, λe=λe, λw=λw, niters=10, tol=1e-9, nepochs=100)
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
            res[:λe] = λe
            res[:λw] = λw


            W = rpt[:W]
            res[:W_nonlinear_mean] = mean(W[:, Nv+1:end])
            res[:W_nonlinear_median] = median(W[:, Nv+1:end])
            res[:W_nonlinear_min] = minimum(W[:, Nv+1:end])
            res[:W_nonlinear_max] = maximum(W[:, Nv+1:end])


            Ŷ = data_reconstruction(mach, df_features)
            res[:rmse_reconst] = rmse(Ŷ, Yorig)

            println("\tReconstruction rmse: ", res[:rmse_reconst])
            println("\tBIC: ", res[:BIC])

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
CSV.write(joinpath("./output/robot-team/", "df_results-Nv_3.csv"), df_res)




unique(df_res.λe)
unique(df_res.λw)
names(df_res)

# Identify the best Nv via BIC, AIC, Q, LLH

df_res[argmax(df_res.Q), [:Q, :Nv, :λe, :λw]]
df_res[argmax(df_res.llhs),  [:llhs, :Nv, :λe, :λw]]
df_res[argmin(df_res.BIC), [:BIC, :Nv, :λe, :λw]]
df_res[argmin(df_res.AIC), [:AIC, :Nv, :λe, :λw]]
df_res[argmin(df_res.rmse_reconst), [:rmse_reconst, :Nv, :λe, :λw, :W_nonlinear_mean, :W_nonlinear_median]]



sort(df_res[df_res.λe .== 0.1, :Q])[1:2]


extrema(Yorig)
mean(Yorig)
median(Yorig)


# train final model:
outpath = "./output/robot-team"
mkpath(outpath)

Nv = 3
k = 75
m = 25
λe = 0.001
λw = 1000.0

json_path = joinpath(outpath, "fit_Nv-$(Nv)_λe-$(λe)_λw-$(λw).json")
save_path = joinpath(outpath, "model_Nv-$(Nv)_λe-$(λe)_λw-$(λw).jls")

gsm = GSMCombo(k=k, m=m, Nv=Nv, λe=λe, λw=λw, niters=25, tol=1e-9, nepochs=1_000)
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
xlims!(ax, 0, 200)
fig

save(joinpath(figpath, "llh.png"), fig)

fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Q (a.u.)", yscale=log10)
lines!(ax, 2:length(res[:Q]), Float64.(res[:Q][2:end]), linewidth=3)
xlims!(0, 200)
fig

save(joinpath(figpath, "Q-fit.png"), fig)




# let's compute the mean values of the weights to see if they are sparse
W = rpt[:W]

fig = Figure();
ax = Axis(fig[1,1], xlabel="λ (nm)", ylabel="Reflectance")
lines!(ax, wavelengths, W[:,1])
lines!(ax, wavelengths, W[:,2])
lines!(ax, wavelengths, W[:,3])

fig




