using HDF5
using CSV, DataFrames
using ProgressMeter

include("./utils/robot-team-config.jl")


datapath = "./data/robot-team"
dye_path = joinpath(datapath, "dye-test")
sup_path = joinpath(datapath, "supervised")

mkpath(datapath)
mkpath(dye_path)
mkpath(sup_path)


h5_basepath = "/Users/johnwaczak/data/robot-team/processed/hsi"
@assert ispath(h5_basepath)


idx_900 = findfirst(wavelengths .≥ 900)


# generate file lists
files_dict = Dict(
    "11-23" => Dict(
        "no-dye" => [
            joinpath(h5_basepath, "11-23", "Scotty_1", "Scotty_1-2.h5"),
            joinpath(h5_basepath, "11-23", "Scotty_1", "Scotty_1-7.h5"),
            joinpath(h5_basepath, "11-23", "Scotty_1", "Scotty_1-13.h5"),
            joinpath(h5_basepath, "11-23", "Scotty_1", "Scotty_1-14.h5"),
            joinpath(h5_basepath, "11-23", "Scotty_2", "Scotty_2-1.h5"),
        ],
        "dye" => [
            joinpath(h5_basepath, "11-23", "Scotty_4", "Scotty_4-1.h5"),
            joinpath(h5_basepath, "11-23", "Scotty_5", "Scotty_5-1.h5"),
        ],
    ),
    "12-09" => Dict(
        "no-dye" => [
            joinpath(h5_basepath, "12-09", "NoDye_1", "NoDye_1-4.h5"),
            joinpath(h5_basepath, "12-09", "NoDye_2", "NoDye_2-2.h5"),
        ],
        "dye" => [
            joinpath(h5_basepath, "12-09", "Dye_1", "Dye_1-6.h5"),
            joinpath(h5_basepath, "12-09", "Dye_2", "Dye_2-5.h5"),
        ],
    ),
    "12-10" => Dict(
        "no-dye" => [
            joinpath(h5_basepath, "12-10", "NoDye_1", "NoDye_1-1.h5"),
            joinpath(h5_basepath, "12-10", "NoDye_2", "NoDye_2-20.h5"),
        ],
        "dye" => [
            joinpath(h5_basepath, "12-10", "Dye_1", "Dye_1-6.h5"),
            joinpath(h5_basepath, "12-10", "Dye_2", "Dye_2-1.h5"),
        ],
    ),
)


# make sure files exist
for (day, collections) in files_dict
    for (collection , files) in collections
        for f in files
            @assert ispath(f)
        end
    end
end


function get_h5_data(h5path, Δx = 0.1, skip_size = 5)
    # open file in read mode
    h5 = h5open(h5path, "r")

    # extract data
    varnames = read(h5["data-Δx_$(Δx)/varnames"])
    Data = read(h5["data-Δx_$(Δx)/Data"])[:, :, :]
    IsInbounds = read(h5["data-Δx_$(Δx)/IsInbounds"])
    Longitudes = read(h5["data-Δx_$(Δx)/Longitudes"])
    Latitudes = read(h5["data-Δx_$(Δx)/Latitudes"])
    xs = read(h5["data-Δx_$(Δx)/X"])
    ys = read(h5["data-Δx_$(Δx)/Y"])

    # close file
    close(h5)

    # generate indices for sampling along grid in x-y space at
    # a spacing given by Δx * skip_size
    IsSkipped = zeros(Bool, size(Data,2), size(Data,3))
    IsSkipped[1:skip_size:end, 1:skip_size:end] .= true

    # only keep pixels within boundary and at skip locations
    ij_inbounds = findall(IsInbounds .&& IsSkipped)

    # create matrices for X,Y coordinates
    X = zeros(size(Data,2), size(Data,3))
    Y = zeros(size(Data,2), size(Data,3))

    # fill with values
    for x_i ∈ axes(Data,2)
        for y_j ∈ axes(Data,3)
            X[x_i, y_j] = xs[x_i]
            Y[x_i, y_j] = ys[y_j]
        end
    end

    # keep only the non-nan pixels
    Data = Data[:, ij_inbounds]

    X = X[ij_inbounds]
    Y = Y[ij_inbounds]
    Longitudes = Longitudes[ij_inbounds]
    Latitudes = Latitudes[ij_inbounds]

    df_h5 = DataFrame(Data', varnames)
    df_h5.x = X
    df_h5.y = Y
    df_h5.longitude = Longitudes
    df_h5.latitude = Latitudes

    return df_h5
end


dfs = []

# loop over files and produce dataframes
for (day, collections) in files_dict
    for (collection , files) in collections
        for f in files
            println("Working on $(f)")
            df = get_h5_data(f)
            push!(dfs, df)
        end
    end
end


df_out = vcat(dfs...);


df_features = df_out[:, 1:idx_900];
df_targets = df_out[:, 463:end];

CSV.write(joinpath(dye_path, "df_features.csv"), df_features)
CSV.write(joinpath(dye_path, "df_targets.csv"), df_targets)


idx_nw = findall(df_targets.NDWI1 .≥ 0.25)

CSV.write(joinpath(dye_path, "df_features-nw.csv"),df_features[idx_nw, :])
CSV.write(joinpath(dye_path, "df_targets-nw.csv"),df_targets[idx_nw, :])


# load in supervised data
df_sup = CSV.read("/Users/johnwaczak/data/robot-team/finalized/Full/df_11_23.csv", DataFrame)


# pinch to desired wavelengths
df_sup_features = df_sup[:, 1:idx_900]
df_sup_targets = df_sup[:, 463:end]


CSV.write(joinpath(sup_path, "df_features.csv"), df_sup_features)
CSV.write(joinpath(sup_path, "df_targets.csv"), df_sup_targets)


# create joined dataset
df_gsm = vcat(df_features[idx_nw,:], df_sup_features)

CSV.write(joinpath(datapath, "df_features.csv"), df_gsm)


