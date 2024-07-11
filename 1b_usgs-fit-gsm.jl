using MLJ, GenerativeTopographicMapping, MLJNonnegativeMatrixFactorization
using DataFrames, CSV
using StableRNGs
using Random
using JSON




datapath = "./data/1_usgs"
outpath = "./output/1_usgs"
@assert ispath(datapath)
@assert ispath(outpath)

include("./utils/fit-metrics.jl")


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

dpath = joinpath(datapath, "linear")

# loop through all SNR and fit GSM model
csvs = [f for f ∈ readdir(dpath) if endswith(f, ".csv")]

snrs = [split(split(splitext(csv)[1], "_")[2], "-")[2] for csv ∈ csvs]
stds = [parse(Float64, split(split(splitext(csv)[1], "_")[3], "-")[2]) for csv ∈ csvs]


k = 75
λs = [0.00001, 0.0001, 0.001, 0.01, 0.1]
Nᵥ = 3



for i ∈ 1:length(csvs)
    csv = csvs[i]
    snr = snrs[i]
    σnoise = stds[i]

    println("\nWorking on SNR = $(snr)")

    # create new folder for each SNR
    opath = joinpath(outpath, "linear", snr)
    if !ispath(opath)
        mkpath(opath)
    end

    # load in data
    df = CSV.read(joinpath(dpath, csv), DataFrame)

    @assert all(Array(df) .≥ 0.0)

    # DO NMF - Euclidean, no regularization
    println("\n------- Euclidean -----\n")
    nmf = NMF(k=Nᵥ, cost=:Euclidean, normalize_abundance=true, tol=1e-9, maxiters=1000, rng=StableRNG(42))
    mach = machine(nmf, df)
    fit!(mach, verbosity=1)

    fp = fitted_params(mach)
    H = fp.H
    W = fp.W

    abund_out = DataFrame(MLJ.transform(mach, df));

    Yorig = Matrix(df)
    Ŷ = (W*H)'

    @assert size(Yorig) == size(Ŷ)

    # generate report
    rpt = report(mach)

    res_dict = Dict()
    res_dict[:cost] = rpt[:cost]
    res_dict[:converged] = rpt[:converged]
    res_dict["σ_orignal"] = σnoise
    res_dict["SNR"] = snr


    res_dict["θ"] = mean([
        minimum([spectral_angle(W[:, idx], R1) for idx ∈ 1:Nᵥ]),
        minimum([spectral_angle(W[:, idx], R2) for idx ∈ 1:Nᵥ]),
        minimum([spectral_angle(W[:, idx], R3) for idx ∈ 1:Nᵥ]),
    ])

    res_dict["RMSE"] = mean([
        minimum([rmse(W[:, idx], R1) for idx ∈ 1:Nᵥ]),
        minimum([rmse(W[:, idx], R2) for idx ∈ 1:Nᵥ]),
        minimum([rmse(W[:, idx], R3) for idx ∈ 1:Nᵥ]),
    ])

    res_dict["SID"] = mean([
        minimum([spectral_information_divergence(W[:, idx], R1) for idx ∈ 1:Nᵥ]),
        minimum([spectral_information_divergence(W[:, idx], R2) for idx ∈ 1:Nᵥ]),
        minimum([spectral_information_divergence(W[:, idx], R3) for idx ∈ 1:Nᵥ]),
    ])

    res_dict["Abundance RMSE"] = mean([
        minimum([rmse(abund_out[:,idx], abund.R1) for idx ∈ 1:3]),
        minimum([rmse(abund_out[:,idx], abund.R2) for idx ∈ 1:3]),
        minimum([rmse(abund_out[:,idx], abund.R3) for idx ∈ 1:3]),
    ])

    res_dict["Reconstruction RMSE"] = rmse(Ŷ, Yorig)

    res_dict["Vertex_1"] = W[:, 1]
    res_dict["Vertex_2"] = W[:, 2]
    res_dict["Vertex_3"] = W[:, 3]

    res_dict["Abundance_1"] = abund_out[:,1]
    res_dict["Abundance_2"] = abund_out[:,2]
    res_dict["Abundance_3"] = abund_out[:,3]

    open(joinpath(opath, "fit-results-nmf-euclidean.json"), "w") do f
        JSON.print(f, res_dict)
    end



    # DO NMF - KL-Divergence, no regularization
    println("\n------- KL -----------\n")
    nmf = NMF(k=Nᵥ, cost=:KL, normalize_abundance=true, tol=1e-9, maxiters=1000, rng=StableRNG(42))
    mach = machine(nmf, df)
    fit!(mach, verbosity=1)

    fp = fitted_params(mach)
    H = fp.H
    W = fp.W

    abund_out = DataFrame(MLJ.transform(mach, df));

    Yorig = Matrix(df)
    Ŷ = (W*H)'

    @assert size(Yorig) == size(Ŷ)

    # generate report
    rpt = report(mach)

    res_dict = Dict()
    res_dict[:cost] = rpt[:cost]
    res_dict[:converged] = rpt[:converged]
    res_dict["σ_orignal"] = σnoise
    res_dict["SNR"] = snr


    res_dict["θ"] = mean([
        minimum([spectral_angle(W[:, idx], R1) for idx ∈ 1:Nᵥ]),
        minimum([spectral_angle(W[:, idx], R2) for idx ∈ 1:Nᵥ]),
        minimum([spectral_angle(W[:, idx], R3) for idx ∈ 1:Nᵥ]),
    ])

    res_dict["RMSE"] = mean([
        minimum([rmse(W[:, idx], R1) for idx ∈ 1:Nᵥ]),
        minimum([rmse(W[:, idx], R2) for idx ∈ 1:Nᵥ]),
        minimum([rmse(W[:, idx], R3) for idx ∈ 1:Nᵥ]),
    ])

    res_dict["SID"] = mean([
        minimum([spectral_information_divergence(W[:, idx], R1) for idx ∈ 1:Nᵥ]),
        minimum([spectral_information_divergence(W[:, idx], R2) for idx ∈ 1:Nᵥ]),
        minimum([spectral_information_divergence(W[:, idx], R3) for idx ∈ 1:Nᵥ]),
    ])

    res_dict["Abundance RMSE"] = mean([
        minimum([rmse(abund_out[:,idx], abund.R1) for idx ∈ 1:3]),
        minimum([rmse(abund_out[:,idx], abund.R2) for idx ∈ 1:3]),
        minimum([rmse(abund_out[:,idx], abund.R3) for idx ∈ 1:3]),
    ])

    res_dict["Reconstruction RMSE"] = rmse(Ŷ, Yorig)

    res_dict["Vertex_1"] = W[:, 1]
    res_dict["Vertex_2"] = W[:, 2]
    res_dict["Vertex_3"] = W[:, 3]

    res_dict["Abundance_1"] = abund_out[:,1]
    res_dict["Abundance_2"] = abund_out[:,2]
    res_dict["Abundance_3"] = abund_out[:,3]

    open(joinpath(opath, "fit-results-nmf-kl.json"), "w") do f
        JSON.print(f, res_dict)
    end



    # DO NMF - L21-Divergence, no regularization
    println("\n------- L21 ----------\n")

    nmf = NMF(k=Nᵥ, cost=:L21, normalize_abundance=true, tol=1e-9, maxiters=1000, rng=StableRNG(42))
    mach = machine(nmf, df)
    fit!(mach, verbosity=1)

    fp = fitted_params(mach)
    H = fp.H
    W = fp.W

    abund_out = DataFrame(MLJ.transform(mach, df));

    Yorig = Matrix(df)
    Ŷ = (W*H)'

    @assert size(Yorig) == size(Ŷ)

    # generate report
    rpt = report(mach)

    res_dict = Dict()
    res_dict[:cost] = rpt[:cost]
    res_dict[:converged] = rpt[:converged]
    res_dict["σ_orignal"] = σnoise
    res_dict["SNR"] = snr


    res_dict["θ"] = mean([
        minimum([spectral_angle(W[:, idx], R1) for idx ∈ 1:Nᵥ]),
        minimum([spectral_angle(W[:, idx], R2) for idx ∈ 1:Nᵥ]),
        minimum([spectral_angle(W[:, idx], R3) for idx ∈ 1:Nᵥ]),
    ])

    res_dict["RMSE"] = mean([
        minimum([rmse(W[:, idx], R1) for idx ∈ 1:Nᵥ]),
        minimum([rmse(W[:, idx], R2) for idx ∈ 1:Nᵥ]),
        minimum([rmse(W[:, idx], R3) for idx ∈ 1:Nᵥ]),
    ])

    res_dict["SID"] = mean([
        minimum([spectral_information_divergence(W[:, idx], R1) for idx ∈ 1:Nᵥ]),
        minimum([spectral_information_divergence(W[:, idx], R2) for idx ∈ 1:Nᵥ]),
        minimum([spectral_information_divergence(W[:, idx], R3) for idx ∈ 1:Nᵥ]),
    ])

    res_dict["Abundance RMSE"] = mean([
        minimum([rmse(abund_out[:,idx], abund.R1) for idx ∈ 1:3]),
        minimum([rmse(abund_out[:,idx], abund.R2) for idx ∈ 1:3]),
        minimum([rmse(abund_out[:,idx], abund.R3) for idx ∈ 1:3]),
    ])

    res_dict["Reconstruction RMSE"] = rmse(Ŷ, Yorig)

    res_dict["Vertex_1"] = W[:, 1]
    res_dict["Vertex_2"] = W[:, 2]
    res_dict["Vertex_3"] = W[:, 3]

    res_dict["Abundance_1"] = abund_out[:,1]
    res_dict["Abundance_2"] = abund_out[:,2]
    res_dict["Abundance_3"] = abund_out[:,3]

    open(joinpath(opath, "fit-results-nmf-L21.json"), "w") do f
        JSON.print(f, res_dict)
    end
end




# GSM
for i ∈ 1:length(csvs)
    csv = csvs[i]
    snr = snrs[i]
    σnoise = stds[i]

    println("\nWorking on SNR = $(snr)")

    # create new folder for each SNR
    opath = joinpath(outpath, "linear", snr)
    if !ispath(opath)
        mkpath(opath)
    end

    # load in data
    df = CSV.read(joinpath(dpath, csv), DataFrame)

    for λ ∈ λs
        println("\tλ = $(λ)\n")
        gsm = GSMLinear(k=k, Nv=Nᵥ, λ=λ, tol=1e-9, nepochs=500, niters=100, rng=StableRNG(42))
        mach = machine(gsm, df)
        fit!(mach, verbosity=1)

        Yorig = Matrix(df)
        Ŷ = data_reconstruction(mach, df)

        # generate report
        rpt = report(mach)

        node_means = rpt[:node_data_means]
        idx_vertices = rpt[:idx_vertices]
        abund_out = DataFrame(MLJ.transform(mach, df));

        res_dict = Dict()
        res_dict[:Q] = rpt[:Q]
        res_dict[:llhs] = rpt[:llhs]
        res_dict[:converged] = rpt[:converged]
        res_dict[:AIC] = rpt[:AIC]
        res_dict[:BIC] = rpt[:BIC]

        res_dict["σ_fit"] = sqrt(rpt[:β⁻¹])
        res_dict["σ_orignal"] = σnoise
        res_dict["SNR"] = snr
        res_dict["λ"] = λ


        res_dict["θ"] = mean([
            minimum([spectral_angle(node_means[:, idx], R1) for idx ∈ idx_vertices]),
            minimum([spectral_angle(node_means[:, idx], R2) for idx ∈ idx_vertices]),
            minimum([spectral_angle(node_means[:, idx], R3) for idx ∈ idx_vertices]),
        ])

        res_dict["RMSE"] = mean([
            minimum([rmse(node_means[:, idx], R1) for idx ∈ idx_vertices]),
            minimum([rmse(node_means[:, idx], R2) for idx ∈ idx_vertices]),
            minimum([rmse(node_means[:, idx], R3) for idx ∈ idx_vertices]),
        ])

        res_dict["SID"] = mean([
            minimum([spectral_information_divergence(node_means[:, idx], R1) for idx ∈ idx_vertices]),
            minimum([spectral_information_divergence(node_means[:, idx], R2) for idx ∈ idx_vertices]),
            minimum([spectral_information_divergence(node_means[:, idx], R3) for idx ∈ idx_vertices]),
        ])

        res_dict["Abundance RMSE"] = mean([
            minimum([rmse(abund_out[:,idx], abund.R1) for idx ∈ 1:3]),
            minimum([rmse(abund_out[:,idx], abund.R2) for idx ∈ 1:3]),
            minimum([rmse(abund_out[:,idx], abund.R3) for idx ∈ 1:3]),
        ])

        res_dict["Reconstruction RMSE"] = rmse(Ŷ, Yorig)

        res_dict["Vertex_1"] = node_means[:, idx_vertices[1]]
        res_dict["Vertex_2"] = node_means[:, idx_vertices[2]]
        res_dict["Vertex_3"] = node_means[:, idx_vertices[3]]

        res_dict["Abundance_1"] = abund_out[:,1]
        res_dict["Abundance_2"] = abund_out[:,2]
        res_dict["Abundance_3"] = abund_out[:,3]

        open(joinpath(opath, "fit-results-gsm_λ-$(λ).json"), "w") do f
            JSON.print(f, res_dict)
        end
    end


    for λ ∈ λs
        println("\tλ = $(λ)\n")

        n_nodes = binomial(k + Nᵥ - 2, Nᵥ -1)

        gsm = GSMBigLinear(n_nodes=n_nodes, Nv=Nᵥ, λ=λ, tol=1e-9, nepochs=500, niters=100, rng=StableRNG(42))
        mach = machine(gsm, df)
        fit!(mach, verbosity=1)

        Yorig = Matrix(df)
        Ŷ = data_reconstruction(mach, df)

        # generate report
        rpt = report(mach)

        node_means = rpt[:node_data_means]
        idx_vertices = rpt[:idx_vertices]
        abund_out = DataFrame(MLJ.transform(mach, df));

        res_dict = Dict()
        res_dict[:Q] = rpt[:Q]
        res_dict[:llhs] = rpt[:llhs]
        res_dict[:converged] = rpt[:converged]
        res_dict[:AIC] = rpt[:AIC]
        res_dict[:BIC] = rpt[:BIC]

        res_dict["σ_fit"] = sqrt(rpt[:β⁻¹])
        res_dict["σ_orignal"] = σnoise
        res_dict["SNR"] = snr
        res_dict["λ"] = λ


        res_dict["θ"] = mean([
            minimum([spectral_angle(node_means[:, idx], R1) for idx ∈ idx_vertices]),
            minimum([spectral_angle(node_means[:, idx], R2) for idx ∈ idx_vertices]),
            minimum([spectral_angle(node_means[:, idx], R3) for idx ∈ idx_vertices]),
        ])

        res_dict["RMSE"] = mean([
            minimum([rmse(node_means[:, idx], R1) for idx ∈ idx_vertices]),
            minimum([rmse(node_means[:, idx], R2) for idx ∈ idx_vertices]),
            minimum([rmse(node_means[:, idx], R3) for idx ∈ idx_vertices]),
        ])

        res_dict["SID"] = mean([
            minimum([spectral_information_divergence(node_means[:, idx], R1) for idx ∈ idx_vertices]),
            minimum([spectral_information_divergence(node_means[:, idx], R2) for idx ∈ idx_vertices]),
            minimum([spectral_information_divergence(node_means[:, idx], R3) for idx ∈ idx_vertices]),
        ])

        res_dict["Abundance RMSE"] = mean([
            minimum([rmse(abund_out[:,idx], abund.R1) for idx ∈ 1:3]),
            minimum([rmse(abund_out[:,idx], abund.R2) for idx ∈ 1:3]),
            minimum([rmse(abund_out[:,idx], abund.R3) for idx ∈ 1:3]),
        ])

        res_dict["Reconstruction RMSE"] = rmse(Ŷ, Yorig)

        res_dict["Vertex_1"] = node_means[:, idx_vertices[1]]
        res_dict["Vertex_2"] = node_means[:, idx_vertices[2]]
        res_dict["Vertex_3"] = node_means[:, idx_vertices[3]]

        res_dict["Abundance_1"] = abund_out[:,1]
        res_dict["Abundance_2"] = abund_out[:,2]
        res_dict["Abundance_3"] = abund_out[:,3]

        open(joinpath(opath, "fit-results-gsm-big_λ-$(λ).json"), "w") do f
            JSON.print(f, res_dict)
        end
    end


end


