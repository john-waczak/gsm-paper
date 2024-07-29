using CairoMakie
using MLJ, GenerativeTopographicMapping
using DataFrames, CSV
# using TernaryDiagrams
using JSON
using Distances



include("./utils/makie-defaults.jl")
set_theme!(mints_theme)
update_theme!(
    figure_padding=30,
    Axis=(
        xticklabelsize=20,
        yticklabelsize=20,
        xlabelsize=20,
        ylabelsize=20,
        titlesize=25,
    ),
    Colorbar=(
        ticklabelsize=20,
        labelsize=22
    )
)

datapath = "./data/1_usgs"
outpath = "./output/1_usgs"
figpath = "./figures/1_usgs"
@assert ispath(datapath)
@assert ispath(outpath)
@assert ispath(figpath)


# load in source spectra
df = CSV.read("./data/src/usgs/usgs.csv", DataFrame)
λs = df.λ
min_to_use = ["Carnallite", "Ammonium Illite", "Biotite"]
R1 = df[:, min_to_use[1]]
R2 = df[:, min_to_use[2]]
R3 = df[:, min_to_use[3]]

abund = CSV.read("./data/1_usgs/df_abund.csv", DataFrame)


# ------------------------
# 1. LINEAR MIXING
# ------------------------

respath = joinpath(outpath, "linear")

results_gsm = []
results_gsm_big = []
results_gsm_combo = []
results_gsm_big_combo = []
results_nmf_euc = []
results_nmf_kl = []
results_nmf_L21 = []

for (root, dirs, files) ∈ walkdir(respath)
    for f ∈ files
        if endswith(f, ".json")
            if occursin("gsm-big-combo_λ", f)
                push!(results_gsm_big_combo, JSON.parsefile(joinpath(root, f)))
            elseif occursin("gsm-combo_λ", f)
                push!(results_gsm_combo, JSON.parsefile(joinpath(root, f)))
            elseif occursin("gsm-big_λ", f)
                push!(results_gsm_big, JSON.parsefile(joinpath(root, f)))
            elseif occursin("gsm_λ", f)
                push!(results_gsm, JSON.parsefile(joinpath(root, f)))
            elseif occursin("nmf-euclidean", f)
                push!(results_nmf_euc, JSON.parsefile(joinpath(root, f)))
            elseif occursin("nmf-kl", f)
                push!(results_nmf_kl, JSON.parsefile(joinpath(root, f)))
            elseif occursin("nmf-L21", f)
                push!(results_nmf_L21, JSON.parsefile(joinpath(root, f)))
            else
                continue
            end
        end
    end
end


df_gsm = Dict[]
df_gsm_big = Dict[]
df_gsm_combo = Dict[]
df_gsm_big_combo = Dict[]
df_nmf_euc = Dict[]
df_nmf_kl = Dict[]
df_nmf_L21 = Dict[]

for i ∈ 1:length(results_gsm)
    d = Dict(
        "Q" => results_gsm[i]["Q"][end],
        "LLH" => results_gsm[i]["llhs"][end],
        "λ" => results_gsm[i]["λ"],
        "BIC" => results_gsm[i]["BIC"],
        "AIC" => results_gsm[i]["AIC"],
        "σ_fit" => results_gsm[i]["σ_fit"],
        "σ_original" => results_gsm[i]["σ_orignal"],
        "SNR" => parse(Float64,results_gsm[i]["SNR"]),
        "SID" => results_gsm[i]["SID"],
        "RMSE" => results_gsm[i]["RMSE"],
        "θ" => results_gsm[i]["θ"],
        "abund_rmse" => results_gsm[i]["Abundance RMSE"],
        "idx" => i,
        "reconst_rmse" => results_gsm[i]["Reconstruction RMSE"]
    )

    push!(df_gsm, d)
end

for i ∈ 1:length(results_gsm_big)
    d = Dict(
        "Q" => results_gsm_big[i]["Q"][end],
        "LLH" => results_gsm_big[i]["llhs"][end],
        "λ" => results_gsm_big[i]["λ"],
        "BIC" => results_gsm_big[i]["BIC"],
        "AIC" => results_gsm_big[i]["AIC"],
        "σ_fit" => results_gsm_big[i]["σ_fit"],
        "σ_original" => results_gsm_big[i]["σ_orignal"],
        "SNR" => parse(Float64,results_gsm_big[i]["SNR"]),
        "SID" => results_gsm_big[i]["SID"],
        "RMSE" => results_gsm_big[i]["RMSE"],
        "θ" => results_gsm_big[i]["θ"],
        "abund_rmse" => results_gsm_big[i]["Abundance RMSE"],
        "idx" => i,
        "reconst_rmse" => results_gsm_big[i]["Reconstruction RMSE"]
    )

    push!(df_gsm_big, d)
end

for i ∈ 1:length(results_gsm_combo)
    d = Dict(
        "Q" => results_gsm_combo[i]["Q"][end],
        "LLH" => results_gsm_combo[i]["llhs"][end],
        "λe" => results_gsm_combo[i]["λe"],
        "λw" => results_gsm_combo[i]["λw"],
        "BIC" => results_gsm_combo[i]["BIC"],
        "AIC" => results_gsm_combo[i]["AIC"],
        "σ_fit" => results_gsm_combo[i]["σ_fit"],
        "σ_original" => results_gsm_combo[i]["σ_orignal"],
        "SNR" => parse(Float64,results_gsm_combo[i]["SNR"]),
        "SID" => results_gsm_combo[i]["SID"],
        "RMSE" => results_gsm_combo[i]["RMSE"],
        "θ" => results_gsm_combo[i]["θ"],
        "abund_rmse" => results_gsm_combo[i]["Abundance RMSE"],
        "idx" => i,
        "reconst_rmse" => results_gsm_combo[i]["Reconstruction RMSE"],
    )

    push!(df_gsm_combo, d)
end

for i ∈ 1:length(results_gsm_big_combo)
    d = Dict(
        "Q" => results_gsm_big_combo[i]["Q"][end],
        "LLH" => results_gsm_big_combo[i]["llhs"][end],
        "λe" => results_gsm_big_combo[i]["λe"],
        "λw" => results_gsm_big_combo[i]["λw"],
        "BIC" => results_gsm_big_combo[i]["BIC"],
        "AIC" => results_gsm_big_combo[i]["AIC"],
        "σ_fit" => results_gsm_big_combo[i]["σ_fit"],
        "σ_original" => results_gsm_big_combo[i]["σ_orignal"],
        "SNR" => parse(Float64,results_gsm_big_combo[i]["SNR"]),
        "SID" => results_gsm_big_combo[i]["SID"],
        "RMSE" => results_gsm_big_combo[i]["RMSE"],
        "θ" => results_gsm_big_combo[i]["θ"],
        "abund_rmse" => results_gsm_big_combo[i]["Abundance RMSE"],
        "idx" => i,
        "reconst_rmse" => results_gsm_big_combo[i]["Reconstruction RMSE"],
    )

    push!(df_gsm_big_combo, d)
end


for i ∈ 1:length(results_nmf_euc)
    d = Dict(
        "cost" => results_nmf_euc[i]["cost"],
        "σ_original" => results_nmf_euc[i]["σ_orignal"],
        "SNR" => parse(Float64,results_nmf_euc[i]["SNR"]),
        "SID" => results_nmf_euc[i]["SID"],
        "RMSE" => results_nmf_euc[i]["RMSE"],
        "θ" => results_nmf_euc[i]["θ"],
        "abund_rmse" => results_nmf_euc[i]["Abundance RMSE"],
        "idx" => i,
        "reconst_rmse" => results_nmf_euc[i]["Reconstruction RMSE"]
    )

    push!(df_nmf_euc, d)
end

for i ∈ 1:length(results_nmf_kl)
    d = Dict(
        "cost" => results_nmf_kl[i]["cost"],
        "σ_original" => results_nmf_kl[i]["σ_orignal"],
        "SNR" => parse(Float64,results_nmf_kl[i]["SNR"]),
        "SID" => results_nmf_kl[i]["SID"],
        "RMSE" => results_nmf_kl[i]["RMSE"],
        "θ" => results_nmf_kl[i]["θ"],
        "abund_rmse" => results_nmf_kl[i]["Abundance RMSE"],
        "idx" => i,
        "reconst_rmse" => results_nmf_kl[i]["Reconstruction RMSE"]
    )

    push!(df_nmf_kl, d)
end

for i ∈ 1:length(results_nmf_L21)
    d = Dict(
        "cost" => results_nmf_L21[i]["cost"],
        "σ_original" => results_nmf_L21[i]["σ_orignal"],
        "SNR" => parse(Float64,results_nmf_L21[i]["SNR"]),
        "SID" => results_nmf_L21[i]["SID"],
        "RMSE" => results_nmf_L21[i]["RMSE"],
        "θ" => results_nmf_L21[i]["θ"],
        "abund_rmse" => results_nmf_L21[i]["Abundance RMSE"],
        "idx" => i,
        "reconst_rmse" => results_nmf_L21[i]["Reconstruction RMSE"]
    )

    push!(df_nmf_L21, d)
end

W = hcat((results_gsm_combo[1]["W"])...)

df_gsm = DataFrame(df_gsm)
df_gsm_big = DataFrame(df_gsm_big)
df_gsm_combo = DataFrame(df_gsm_combo)
df_gsm_big_combo = DataFrame(df_gsm_big_combo)
df_nmf_euc = DataFrame(df_nmf_euc)
df_nmf_kl = DataFrame(df_nmf_kl)
df_nmf_L21 = DataFrame(df_nmf_L21)


# df_gsm_combo = df_gsm_combo[.!(isnothing.(df_gsm_combo.θ)), :]
# df_gsm_big_combo= df_gsm_big_combo[.!(isnothing.(df_gsm_big_combo.θ)), :]


CSV.write(joinpath(respath, "df_gsm.csv"), df_gsm)
CSV.write(joinpath(respath, "df_gsm_big.csv"), df_gsm_big)
CSV.write(joinpath(respath, "df_gsm_combo.csv"), df_gsm_combo)
CSV.write(joinpath(respath, "df_gsm_big_combo.csv"), df_gsm_big_combo)
CSV.write(joinpath(respath, "df_nmf_euc.csv"), df_nmf_euc)
CSV.write(joinpath(respath, "df_nmf_kl.csv"), df_nmf_kl)
CSV.write(joinpath(respath, "df_nmf_L21.csv"), df_nmf_L21)



# let's just take λ = 0.001
df_gsm_plot = df_gsm[df_gsm.λ .== 0.001, :]
df_gsm_big_plot = df_gsm_big[df_gsm_big.λ .== 0.001, :]
df_gsm_combo_plot = df_gsm_combo[df_gsm_combo.λe .== 0.001 .&& df_gsm_combo.λw .== 100.0, :]
df_gsm_big_combo_plot = df_gsm_big_combo[df_gsm_big_combo.λe .== 0.001 .&& df_gsm_big_combo.λw .== 100.0, :]


# sort everything by SNR
sort!(df_gsm_plot, :SNR, rev=true)
sort!(df_gsm_big_plot, :SNR, rev=true)
sort!(df_gsm_combo_plot, :SNR, rev=true)
sort!(df_gsm_big_combo_plot, :SNR, rev=true)
sort!(df_nmf_euc, :SNR, rev=true)
sort!(df_nmf_kl, :SNR, rev=true)
sort!(df_nmf_L21, :SNR, rev=true)


df_gsm_plot[:, [:SNR, :θ, :abund_rmse, :reconst_rmse]]
df_gsm_combo_plot[:, [:SNR, :θ, :abund_rmse, :reconst_rmse]]
df_gsm_big_combo_plot[:, [:SNR, :θ, :abund_rmse, :reconst_rmse]]


# figure out the lambda with the best θ for each SNR
fig = Figure(;);
gl = fig[1,1] = GridLayout();
ax = Axis(gl[2,1], xlabel="SNR (dB)", ylabel="Mean Spectral Angle (degrees)", xticks=(1:9, ["∞", "35", "30", "25", "20", "15", "10", "5", "0"]), xminorgridvisible=false, yminorgridvisible=false);

l_gsm_combo = scatter!(ax, 1:9, df_gsm_combo_plot[:, :θ], markersize=15, color=:green)
lines!(ax, 1:9, df_gsm_combo_plot[:, :θ], linewidth=3, linestyle=:dash, color=:green)

l_gsm_big_combo = scatter!(ax, 1:9, df_gsm_big_combo_plot[:, :θ], markersize=15, color=mints_colors[1], marker=:utriangle)
lines!(ax, 1:9, df_gsm_big_combo_plot[:, :θ], linewidth=3, linestyle=:dash, color=mints_colors[1])

l_nmf_euc = scatter!(ax, 1:9, df_nmf_euc[:, :θ], markersize=15, color=mints_colors[2], marker=:diamond)
lines!(ax, 1:9, df_nmf_euc[:, :θ], linewidth=3, linestyle=:dash, color=mints_colors[2])

l_nmf_kl = scatter!(ax, 1:9, df_nmf_kl[:, :θ], markersize=15, color=:red, marker=:cross)
lines!(ax, 1:9, df_nmf_kl[:, :θ], linewidth=3, linestyle=:dash, color=:red)

l_nmf_L21 = scatter!(ax, 1:9, df_nmf_L21[:, :θ], markersize=15, color=:orange, marker=:star5)
lines!(ax, 1:9, df_nmf_L21[:, :θ], linewidth=3, linestyle=:dash, color=:orange)

Leg = Legend(gl[1,1], [l_gsm_combo, l_gsm_big_combo, l_nmf_euc, l_nmf_kl, l_nmf_L21], ["GSM", "GSM (Big)", "NMF (ℓ₂)", "NMF (KL)", "NMF (ℓ₂,₁)"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=14, height=-5)

xlims!(ax, 0, 9.2)
fig
# ax2 = Axis(gl[2,2])
# hidedecorations!(ax2)
# hidespines!(ax2)
# linkyaxes!(ax, ax2)

# colsize!(gl, 2, Relative(0.1))

fig

save(joinpath(figpath, "linear", "spectral-angle-vs-snr.png"), fig)



fig = Figure(; px_per_unit=30);
ax = Axis(fig[2,1], xlabel="SNR (dB)", ylabel="Mean RMSE", xticks=(1:9, ["∞", "35", "30", "25", "20", "15", "10", "5", "0"]), xminorgridvisible=false, yminorgridvisible=false);

l_gsm_combo = scatter!(ax, 1:9, df_gsm_combo_plot[:, :RMSE], markersize=15, color=:green)
lines!(ax, 1:9, df_gsm_combo_plot[:, :RMSE], linewidth=3, linestyle=:dash, color=:green)

l_gsm_big_combo = scatter!(ax, 1:9, df_gsm_big_combo_plot[:, :RMSE], markersize=15, color=mints_colors[1], marker=:utriangle)
lines!(ax, 1:9, df_gsm_big_combo_plot[:, :RMSE], linewidth=3, linestyle=:dash, color=mints_colors[1])

l_nmf_euc = scatter!(ax, 1:9, df_nmf_euc[:, :RMSE], markersize=15, color=mints_colors[2], marker=:diamond)
lines!(ax, 1:9, df_nmf_euc[:, :RMSE], linewidth=3, linestyle=:dash, color=mints_colors[2])

l_nmf_kl = scatter!(ax, 1:9, df_nmf_kl[:, :RMSE], markersize=15, color=:red, marker=:cross)
lines!(ax, 1:9, df_nmf_kl[:, :RMSE], linewidth=3, linestyle=:dash, color=:red)

l_nmf_L21 = scatter!(ax, 1:9, df_nmf_L21[:, :RMSE], markersize=15, color=:orange, marker=:star5)
lines!(ax, 1:9, df_nmf_L21[:, :RMSE], linewidth=3, linestyle=:dash, color=:orange)

fig[1,1] = Legend(fig, [l_gsm_combo, l_gsm_big_combo, l_nmf_euc, l_nmf_kl, l_nmf_L21], ["GSM", "GSM (Big)", "NMF (ℓ₂)", "NMF (KL)", "NMF (ℓ₂,₁)"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=14, height=-5)
fig

save(joinpath(figpath, "linear", "rmse-vs-snr.png"), fig)



fig = Figure();
ax = Axis(fig[2,1], xlabel="SNR (dB)", ylabel="Mean Spectral Information Divergence", xticks=(1:9, ["∞", "35", "30", "25", "20", "15", "10", "5", "0"]), xminorgridvisible=false, yminorgridvisible=false);

l_gsm_combo = scatter!(ax, 1:9, df_gsm_combo_plot[:, :SID], markersize=15, color=:green)
lines!(ax, 1:9, df_gsm_combo_plot[:, :SID], linewidth=3, linestyle=:dash, color=:green)

l_gsm_big_combo = scatter!(ax, 1:9, df_gsm_big_combo_plot[:, :SID], markersize=15, color=mints_colors[1], marker=:utriangle)
lines!(ax, 1:9, df_gsm_big_combo_plot[:, :SID], linewidth=3, linestyle=:dash, color=mints_colors[1])

l_nmf_euc = scatter!(ax, 1:9, df_nmf_euc[:, :SID], markersize=15, color=mints_colors[2], marker=:diamond)
lines!(ax, 1:9, df_nmf_euc[:, :SID], linewidth=3, linestyle=:dash, color=mints_colors[2])

l_nmf_kl = scatter!(ax, 1:9, df_nmf_kl[:, :SID], markersize=15, color=:red, marker=:cross)
lines!(ax, 1:9, df_nmf_kl[:, :SID], linewidth=3, linestyle=:dash, color=:red)

l_nmf_L21 = scatter!(ax, 1:9, df_nmf_L21[:, :SID], markersize=15, color=:orange, marker=:star5)
lines!(ax, 1:9, df_nmf_L21[:, :SID], linewidth=3, linestyle=:dash, color=:orange)

fig[1,1] = Legend(fig, [l_gsm_combo, l_gsm_big_combo, l_nmf_euc, l_nmf_kl, l_nmf_L21], ["GSM", "GSM (Big)", "NMF (ℓ₂)", "NMF (KL)", "NMF (ℓ₂,₁)"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=14, height=-5)

fig

save(joinpath(figpath, "linear", "sid-vs-snr.png"), fig)


fig = Figure();
ax = Axis(fig[2,1], xlabel="SNR (dB)", ylabel="Data Reconstruction RMSE", xticks=(1:9, ["∞", "35", "30", "25", "20", "15", "10", "5", "0"]), xminorgridvisible=false, yminorgridvisible=true); #, yscale=log10);

l_gsm_combo = scatter!(ax, 1:9, df_gsm_combo_plot[:, :reconst_rmse], markersize=15, color=:green)
lines!(ax, 1:9, df_gsm_combo_plot[:, :reconst_rmse], linewidth=3, linestyle=:dash, color=:green)

l_gsm_big_combo = scatter!(ax, 1:9, df_gsm_big_combo_plot[:, :reconst_rmse], markersize=15, color=mints_colors[1], marker=:utriangle)
lines!(ax, 1:9, df_gsm_big_combo_plot[:, :reconst_rmse], linewidth=3, linestyle=:dash, color=mints_colors[1])

l_nmf_euc = scatter!(ax, 1:9, df_nmf_euc[:, :reconst_rmse], markersize=15, color=mints_colors[2], marker=:diamond)
lines!(ax, 1:9, df_nmf_euc[:, :reconst_rmse], linewidth=3, linestyle=:dash, color=mints_colors[2])

l_nmf_kl = scatter!(ax, 1:9, df_nmf_kl[:, :reconst_rmse], markersize=15, color=:red, marker=:cross)
lines!(ax, 1:9, df_nmf_kl[:, :reconst_rmse], linewidth=3, linestyle=:dash, color=:red)

l_nmf_L21 = scatter!(ax, 1:9, df_nmf_L21[:, :reconst_rmse], markersize=15, color=:orange, marker=:star5)
lines!(ax, 1:9, df_nmf_L21[:, :reconst_rmse], linewidth=3, linestyle=:dash, color=:orange)

fig[1,1] = Legend(fig, [l_gsm_combo, l_gsm_big_combo, l_nmf_euc, l_nmf_kl, l_nmf_L21], ["GSM", "GSM (Big)", "NMF (ℓ₂)", "NMF (KL)", "NMF (ℓ₂,₁)"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=14, height=-5)

fig

save(joinpath(figpath, "linear", "reconstruction-rmse.png"), fig)



fig = Figure();
ax = Axis(fig[2,1], xlabel="SNR (dB)", ylabel="Mean Abundance RMSE", xticks=(1:9, ["∞", "35", "30", "25", "20", "15", "10", "5", "0"]), xminorgridvisible=false, yminorgridvisible=false);

l_gsm_combo = scatter!(ax, 1:9, df_gsm_combo_plot[:, :abund_rmse], markersize=15, color=:green)
lines!(ax, 1:9, df_gsm_combo_plot[:, :abund_rmse], linewidth=3, linestyle=:dash, color=:green)

l_gsm_big_combo = scatter!(ax, 1:9, df_gsm_big_combo_plot[:, :abund_rmse], markersize=15, color=mints_colors[1], marker=:utriangle)
lines!(ax, 1:9, df_gsm_big_combo_plot[:, :abund_rmse], linewidth=3, linestyle=:dash, color=mints_colors[1])

l_nmf_euc = scatter!(ax, 1:9, df_nmf_euc[:, :abund_rmse], markersize=15, color=mints_colors[2], marker=:diamond)
lines!(ax, 1:9, df_nmf_euc[:, :abund_rmse], linewidth=3, linestyle=:dash, color=mints_colors[2])

l_nmf_kl = scatter!(ax, 1:9, df_nmf_kl[:, :abund_rmse], markersize=15, color=:red, marker=:cross)
lines!(ax, 1:9, df_nmf_kl[:, :abund_rmse], linewidth=3, linestyle=:dash, color=:red)

l_nmf_L21 = scatter!(ax, 1:9, df_nmf_L21[:, :abund_rmse], markersize=15, color=:orange, marker=:star5)
lines!(ax, 1:9, df_nmf_L21[:, :abund_rmse], linewidth=3, linestyle=:dash, color=:orange)

fig[1,1] = Legend(fig, [l_gsm_combo, l_gsm_big_combo, l_nmf_euc, l_nmf_kl, l_nmf_L21], ["GSM", "GSM (Big)", "NMF (ℓ₂)", "NMF (KL)", "NMF (ℓ₂,₁)"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=14, height=-5)

fig

save(joinpath(figpath, "linear", "abundance-rmse-vs-snr.png"), fig)



# create plots for linear data
idx_plot = df_gsm_combo_plot[df_gsm_combo_plot.SNR .== 20, :].idx


res = results_gsm_combo[idx_plot][1]

A1_fit = Float64.(res["Abundance_1"])
A2_fit = Float64.(res["Abundance_2"])
A3_fit = Float64.(res["Abundance_3"])
vertex_1 = Float64.(res["Vertex_1"])
vertex_2 = Float64.(res["Vertex_2"])
vertex_3 = Float64.(res["Vertex_3"])
σ_fit = res["σ_fit"]
σ_orig = res["σ_orignal"]
λs = df.λ

# plot log likelihoods
fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Log-likelihood", yscale=log10)
lines!(ax, 2:length(res["llhs"]), Float64.(res["llhs"][2:end]), linewidth=3)
xlims!(ax, 0, 100)
fig

save(joinpath(figpath, "linear", "llh.png"), fig)

fig = Figure();
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="Q (a.u.)", yscale=log10)
lines!(ax, 2:length(res["Q"]), Float64.(res["Q"][2:end]), linewidth=3)
xlims!(0, 100)
fig

save(joinpath(figpath, "linear", "Q-fit.png"), fig)


# plot endmembers for linear data
fig = Figure(; size=(900, 500),  px_per_unit=30);
ax = Axis(fig[1,1], xlabel="λ (μm)", ylabel="Reflectance");

l1 = lines!(ax, λs, R1, color=mints_colors[2], linewidth=2, linestyle=:dash)
l2 = lines!(ax, λs, R2, color=mints_colors[1], linewidth=2, linestyle=:dash)
l3 = lines!(ax, λs, R3, color=mints_colors[3], linewidth=2, linestyle=:dash)


# vertex 1 -> green, vertex 2 -> blue, vertex 3 -> red
band!(ax, λs, vertex_3 .- (2*σ_fit), vertex_3 .+ (2*σ_fit), color=(mints_colors[2], 0.35))
l1_fit = lines!(ax, λs, vertex_3, color=(mints_colors[2], 0.75), linewidth=2) #, linestyle=:dash)

band!(ax, λs, vertex_1 .- (2*σ_fit), vertex_1 .+ (2*σ_fit), color=(mints_colors[1], 0.35))
l2_fit = lines!(ax, λs, vertex_1, color=(mints_colors[1], 0.75), linewidth=2) #, linestyle=:dash)

band!(ax, λs, vertex_2 .- (2*σ_fit), vertex_2 .+ (2*σ_fit), color=(mints_colors[3], 0.35))
l3_fit = lines!(ax, λs, vertex_2, color=(mints_colors[3], 0.75), linewidth=2) #, linestyle=:dash)

fig[1,2] = Legend(fig, [l1, l2, l3, l1_fit, l2_fit, l3_fit], [min_to_use..., "Vertex 1", "Vertex 2", "Vertex 3"], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=13, height=-5)
xlims!(ax, λs[1], λs[end])
ylims!(ax, 0, 1)
fig

save(joinpath(figpath, "linear", "extracted-endmembers.png"), fig)
