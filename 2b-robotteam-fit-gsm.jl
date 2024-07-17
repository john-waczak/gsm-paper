using MLJ, GenerativeTopographicMapping
using DataFrames, CSV
using StableRNGs
using Random
using JSON
using JLSO
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



# Based on this, between 4-8 endmembers should be fine
n_nodes = 3000
n_rbfs = 250


#k = 25 # this roughly corresponds to 3000 nodes
# Nv_s = [3, 4, 5, 6, 7, 8, 9]


Nv_s = [3, 4, 5, 6, 7]
s = 1/10
λe_s = [0.01, 0.001, 0.0001]
λw_s = [10_000, 1_000, 100, 10, 1]

println("\tNumber of models to train: ", length(Nv_s) * length(λe_s) * length(λw_s))
println("\tTraiing models...")

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

                gsm = GSMBigCombo(n_nodes=n_nodes, n_rbfs=n_rbfs, s=s, Nv=Nv, niters=10, tol=1e-9, nepochs=100)

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

                # Save the machine for later use
                # https://juliaai.github.io/MLJ.jl/stable/machines/

                smach = serializable(mach)
                JLSO.save(jlso_path, :machine => smach)
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





# Identify the best Nv via BIC, AIC, Q, LLH
gdf = groupby(df_res, :Nv)

df_Q = combine(gdf, :Q => maximum)
df_llh = combine(gdf, :llhs => maximum)
df_BIC = combine(gdf, :BIC => minimum)
df_AIC = combine(gdf, :AIC => minimum)

Nv_Q = df_Q[argmax(df_Q.Q_maximum), :Nv]
Nv_llh = df_llh[argmax(df_llh.llhs_maximum), :Nv]
Nv_BIC = df_BIC[argmin(df_BIC.BIC_minimum), :Nv]
Nv_AIC = df_AIC[argmin(df_AIC.AIC_minimum), :Nv]


