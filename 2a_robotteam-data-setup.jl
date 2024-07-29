using HDF5
using CSV, DataFrames
using ProgressMeter
using Random, StableRNGs

include("./utils/robot-team-config.jl")
datapath = "./data/robot-team"
if !ispath(datapath)
    mkpath(datapath)
end

# load the model
h5_path = joinpath(datapath, "sample-datacube.h5")
@assert ispath(h5_path)


# open HDF5 file and gather data
h5 = h5open(h5_path, "r")
Δx = 0.1
varnames = read(h5["data-Δx_$(Δx)/varnames"]);
Data = read(h5["data-Δx_$(Δx)/Data"])[:, :, :];
IsInbounds = read(h5["data-Δx_$(Δx)/IsInbounds"]);
Longitudes = read(h5["data-Δx_$(Δx)/Longitudes"]);
Latitudes = read(h5["data-Δx_$(Δx)/Latitudes"]);
xs = read(h5["data-Δx_$(Δx)/X"]);
ys = read(h5["data-Δx_$(Δx)/Y"]);
close(h5)

# get X and Y positions
X = zeros(size(Data,2), size(Data,3))
Y = zeros(size(Data,2), size(Data,3))

# fill with values
for x_i ∈ axes(Data,2)
    for y_j ∈ axes(Data,3)
        X[x_i, y_j] = xs[x_i]
        Y[x_i, y_j] = ys[y_j]
    end
end

idx_NDWI = findfirst(varnames .== "NDWI1")
idx_900 = findfirst(wavelengths .≥ 900)

IsPlume = zeros(Bool, size(IsInbounds))
IsPlume[1:250, 150:end] .= true

ij_nw = findall(IsInbounds .&& (Data[idx_NDWI,:,:] .> 0.25) .&& .!IsPlume)
ij_plume = findall(IsPlume .&& IsInbounds)


length(ij_nw)     # 105_269
length(ij_plume)  # 46_181
# @assert !any([ij in ij_nw for ij in ij_plume])

idx_out = vcat(
    ij_nw[randperm(StableRNG(42), length(ij_nw))[1:10_000]],
    ij_plume[randperm(StableRNG(42), length(ij_plume))[1:5_000]]
)

df_out = DataFrame(Data[1:idx_900,idx_out]', varnames[1:idx_900])

CSV.write(joinpath(datapath, "df_features.csv"), df_out)




# dye_path = joinpath(datapath, "dye-test")
# sup_path = joinpath(datapath, "supervised")

# mkpath(datapath)
# mkpath(dye_path)
# mkpath(sup_path)


# h5_basepath = "/Users/johnwaczak/data/robot-team/processed/hsi"
# @assert ispath(h5_basepath)


# idx_900 = findfirst(wavelengths .≥ 900)


# # generate file lists
# files_dict = Dict(
#     "11-23" => Dict(
#         "no-dye" => [
#             joinpath(h5_basepath, "11-23", "Scotty_1", "Scotty_1-2.h5"),
#             joinpath(h5_basepath, "11-23", "Scotty_1", "Scotty_1-7.h5"),
#             joinpath(h5_basepath, "11-23", "Scotty_1", "Scotty_1-13.h5"),
#             joinpath(h5_basepath, "11-23", "Scotty_1", "Scotty_1-14.h5"),
#             joinpath(h5_basepath, "11-23", "Scotty_2", "Scotty_2-1.h5"),
#         ],
#         "dye" => [
#             joinpath(h5_basepath, "11-23", "Scotty_4", "Scotty_4-1.h5"),
#             joinpath(h5_basepath, "11-23", "Scotty_5", "Scotty_5-1.h5"),
#         ],
#     ),
#     "12-09" => Dict(
#         "no-dye" => [
#             joinpath(h5_basepath, "12-09", "NoDye_1", "NoDye_1-4.h5"),
#             joinpath(h5_basepath, "12-09", "NoDye_2", "NoDye_2-2.h5"),
#         ],
#         "dye" => [
#             joinpath(h5_basepath, "12-09", "Dye_1", "Dye_1-6.h5"),
#             joinpath(h5_basepath, "12-09", "Dye_2", "Dye_2-5.h5"),
#         ],
#     ),
#     "12-10" => Dict(
#         "no-dye" => [
#             joinpath(h5_basepath, "12-10", "NoDye_1", "NoDye_1-1.h5"),
#             joinpath(h5_basepath, "12-10", "NoDye_2", "NoDye_2-20.h5"),
#         ],
#         "dye" => [
#             joinpath(h5_basepath, "12-10", "Dye_1", "Dye_1-6.h5"),
#             joinpath(h5_basepath, "12-10", "Dye_2", "Dye_2-1.h5"),
#         ],
#     ),
# )


# # make sure files exist
# for (day, collections) in files_dict
#     for (collection , files) in collections
#         for f in files
#             @assert ispath(f)
#         end
#     end
# end


# function get_h5_data(h5path, Δx = 0.1, skip_size = 5)
#     # open file in read mode
#     h5 = h5open(h5path, "r")

#     # extract data
#     varnames = read(h5["data-Δx_$(Δx)/varnames"])
#     Data = read(h5["data-Δx_$(Δx)/Data"])[:, :, :]
#     IsInbounds = read(h5["data-Δx_$(Δx)/IsInbounds"])
#     Longitudes = read(h5["data-Δx_$(Δx)/Longitudes"])
#     Latitudes = read(h5["data-Δx_$(Δx)/Latitudes"])
#     xs = read(h5["data-Δx_$(Δx)/X"])
#     ys = read(h5["data-Δx_$(Δx)/Y"])

#     # close file
#     close(h5)

#     # generate indices for sampling along grid in x-y space at
#     # a spacing given by Δx * skip_size
#     IsSkipped = zeros(Bool, size(Data,2), size(Data,3))
#     IsSkipped[1:skip_size:end, 1:skip_size:end] .= true

#     # only keep pixels within boundary and at skip locations
#     ij_inbounds = findall(IsInbounds .&& IsSkipped)

#     # create matrices for X,Y coordinates
#     X = zeros(size(Data,2), size(Data,3))
#     Y = zeros(size(Data,2), size(Data,3))

#     # fill with values
#     for x_i ∈ axes(Data,2)
#         for y_j ∈ axes(Data,3)
#             X[x_i, y_j] = xs[x_i]
#             Y[x_i, y_j] = ys[y_j]
#         end
#     end

#     # keep only the non-nan pixels
#     Data = Data[:, ij_inbounds]

#     X = X[ij_inbounds]
#     Y = Y[ij_inbounds]
#     Longitudes = Longitudes[ij_inbounds]
#     Latitudes = Latitudes[ij_inbounds]

#     df_h5 = DataFrame(Data', varnames)
#     df_h5.x = X
#     df_h5.y = Y
#     df_h5.longitude = Longitudes
#     df_h5.latitude = Latitudes

#     return df_h5
# end


# dfs_dye = []
# dfs_nodye = []


# # loop over files and produce dataframes
# for (day, collections) in files_dict
#     for (collection , files) in collections
#         for f in files
#             if collection == "dye"
#                 println("Working on $(f)")
#                 df = get_h5_data(f)
#                 push!(dfs_dye, df)
#             else
#                 println("Working on $(f)")
#                 df = get_h5_data(f)
#                 push!(dfs_nodye, df)
#             end
#         end
#     end
# end


# df_dye = vcat(dfs_dye...);
# df_nodye = vcat(dfs_nodye...);

# # keep only water pixels
# idx_nw_dye = findall(df_dye.NDWI1 .≥ 0.25)
# idx_nw_nodye = findall(df_nodye.NDWI1 .≥ 0.25)

# df_dye_features = df_dye[idx_nw_dye, 1:idx_900]
# df_dye_targets = df_dye[idx_nw_dye, 463:end]

# df_nodye_features = df_nodye[idx_nw_nodye, 1:idx_900]
# df_nodye_targets = df_nodye[idx_nw_nodye, 463:end]


# # save the data
# CSV.write(joinpath(dye_path, "df_dye_features.csv"), df_dye_features)
# CSV.write(joinpath(dye_path, "df_dye_targets.csv"), df_dye_targets)
# CSV.write(joinpath(dye_path, "df_nodye_features.csv"), df_nodye_features)
# CSV.write(joinpath(dye_path, "df_nodye_targets.csv"), df_nodye_targets)


# # load in supervised data
# df_sup = CSV.read("/Users/johnwaczak/data/robot-team/finalized/Full/df_11_23.csv", DataFrame)

# # pinch to desired wavelengths
# idx_nw = findall(df_sup.NDWI1 .≥ 0.25)
# df_sup_features = df_sup[idx_nw, 1:idx_900]
# df_sup_targets = df_sup[idx_nw, 463:end]


# CSV.write(joinpath(sup_path, "df_features.csv"), df_sup_features)
# CSV.write(joinpath(sup_path, "df_targets.csv"), df_sup_targets)


# # create joined dataset with 20_000 total datapoints
# nrow(df_dye_features) + nrow(df_sup_features)

# using Random
# using StableRNGs

# rng = StableRNG(42)
# idx_dye = shuffle(rng, 1:nrow(df_dye_features))[1:10_000]
# idx_sup = shuffle(rng, 1:nrow(df_sup_features))[1:10_000]

# df_gsm = vcat(df_dye_features[idx_dye,:], df_sup_features[idx_sup, :])
# CSV.write(joinpath(datapath, "df_features.csv"), df_gsm)


