include("./utils/background-viz.jl")

# see https://en.wikipedia.org/wiki/Geographic_coordinate_system#Latitude_and_longitude
# for detail
ϕ_scale = 33.70093
m_per_deg = 111412.84*cosd(ϕ_scale) - 93.5*cosd(3*ϕ_scale) + 0.118 * cosd(5*ϕ_scale)
λ_scale_l = -97.7166
λ_scale_r = λ_scale_l + 30/m_per_deg

w= -97.717472
n= 33.703572
s= 33.700797
e= -97.712413

satmap = get_background_satmap(w,e,s,n)

lon_min, lon_max = (-97.7168, -97.7125)
lat_min, lat_max = (33.70075, 33.7035)


using MLJ, GenerativeTopographicMapping
using DataFrames, CSV
using StableRNGs
using Random
using JSON
using LinearAlgebra
using HDF5
using Statistics

include("./utils/robot-team-config.jl")
include("./utils/datacube-viz.jl")
include("./utils/fit-metrics.jl")

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




figpath = "./figures/robot-team"
outpath = "./output/robot-team"
json_path = joinpath(outpath, "fit_final.json")
mdl_path = joinpath(outpath, "model_final.jls")
mach = machine(mdl_path)


f_list_dye = [
    joinpath("./data", "robot-team", "dye_1.h5"),
    joinpath("./data", "robot-team", "dye_21.h5"),
    joinpath("./data", "robot-team", "dye_22.h5"),
]
@assert all(ispath.(f_list_dye))


fpath_1 = joinpath("./data", "robot-team", "sample-datacube.h5")
fpath_2 = joinpath("./data", "robot-team", "sample-datacube-2.h5")
fpath_3 = joinpath("./data", "robot-team", "sample-datacube-3.h5")


idx_900 = findfirst(wavelengths .≥ 900)
function get_data_for_heatmap(h5path, Δx = 0.1,)
    h5 = h5open(h5path, "r")

    # extract data
    varnames = read(h5["data-Δx_$(Δx)/varnames"])
    Data = read(h5["data-Δx_$(Δx)/Data"])[:, :, :]
    IsInbounds = read(h5["data-Δx_$(Δx)/IsInbounds"])
    Longitudes = read(h5["data-Δx_$(Δx)/Longitudes"])
    Latitudes = read(h5["data-Δx_$(Δx)/Latitudes"])
    xs = read(h5["data-Δx_$(Δx)/X"])
    ys = read(h5["data-Δx_$(Δx)/Y"])
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
    ij_inbounds  = findall(IsInbounds .&& (Data[idx_NDWI,:,:] .> 0.25))
    ij_inbounds  = findall(IsInbounds)

    # keep only the non-nan pixels
    Data = Data[1:idx_900, :, :]

    return Data, X, Y, Longitudes, Latitudes, ij_inbounds, IsInbounds, varnames
end



R1, X1, Y1, Lon1, Lat1, ij_inbounds1, IsInbounds1, varnames = get_data_for_heatmap(fpath_1);
R2, X2, Y2, Lon2, Lat2, ij_inbounds2, IsInbounds2, varnames = get_data_for_heatmap(fpath_2);
R3, X3, Y3, Lon3, Lat3, ij_inbounds3, IsInbounds3, varnames = get_data_for_heatmap(fpath_3);

Zpred1 = fill(NaN, size(X1));
Zpred2 = fill(NaN, size(X2));
Zpred3 = fill(NaN, size(X3));


let
    println("Generating Data Frame")

    IsPlume = zeros(Bool, size(IsInbounds1))
    IsPlume[1:250, 150:end] .= true
    ij_plume = findall(IsPlume .&& IsInbounds1)
    df_plume = DataFrame(R1[:,ij_plume]', varnames[1:idx_900])
    println("Getting abundances")
    df_z_plume = DataFrame(MLJ.transform(mach, df_plume))

    println("Filling output array")
    Zpred1[ij_plume] .= df_z_plume.Z3
end

let
    println("Generating Data Frame")

    IsPlume = zeros(Bool, size(IsInbounds2))
    IsPlume[1:330, 80:end] .= true
    ij_plume = findall(IsPlume .&& IsInbounds2)
    df_plume = DataFrame(R2[:,ij_plume]', varnames[1:idx_900])
    println("Getting abundances")
    df_z_plume = DataFrame(MLJ.transform(mach, df_plume))

    println("Filling output array")
    Zpred2[ij_plume] .= df_z_plume.Z3
end

let
    println("Generating Data Frame")

    IsPlume = zeros(Bool, size(IsInbounds3))
    IsPlume[1:350, 1:200] .= true
    ij_plume = findall(IsPlume .&& IsInbounds3)
    df_plume = DataFrame(R3[:,ij_plume]', varnames[1:idx_900])
    println("Getting abundances")
    df_z_plume = DataFrame(MLJ.transform(mach, df_plume))

    println("Filling output array")
    Zpred3[ij_plume] .= df_z_plume.Z3
end




lon_min, lon_max = (-97.7168, -97.7155)
lat_min, lat_max = (33.7015, 33.70275)

lon_l, lon_h = extrema(Lon1)
lat_l, lat_h = extrema(Lat1)

# set up color map
c_high = colorant"#f73a2d"
c_low = colorant"#450a06"
cmap = cgrad([c_low, c_high])
lim_low = 0.3
lim_high = 1.0
clims = (lim_low, lim_high)

area_pts = [z for z ∈ Zpred1[1:250, 150:end] if !isnan(z) .&& z .> lim_low]
tot_area = length(area_pts) * (0.1^2)

fig = Figure();
ax = CairoMakie.Axis(
    fig[1,1],
    xlabel="Longitude", xtickformat = x -> string.(round.(x .+ lon_min, digits=6)), xticklabelsize= 14,
    xticksize=7,
    xminorticksvisible=true,
    ylabel="Latitude",  ytickformat = y -> string.(round.(y .+ lat_min, digits=6)), yticklabelsize = 14,
    yticksize=7,
    yminorticksvisible=true,
    title="Total Area = $(round(tot_area, digits=1)) m²",
    titlealign=:left,
    titlefont=:regular,
    titlesize=13,
);
bg = heatmap!(
    ax,
    (satmap.w - lon_min)..(satmap.e - lon_min),
    (satmap.s - lat_min)..(satmap.n - lat_min),
    satmap.img,
    alpha=0.65,
)

h = heatmap!(ax, (lon_l-lon_min)..(lon_h-lon_min), (lat_l-lat_min)..(lat_h-lat_min), Zpred1, colormap=cmap, colorrange=clims, lowclip=:transparent)

Δϕ = 33.7015 - 33.70075
xlims!(ax, 0, lon_max - lon_min)
ylims!(ax, 0, lat_max - lat_min)

# add 30 meter scale bar
lines!(ax, [λ_scale_l - lon_min, λ_scale_r - lon_min], [ϕ_scale + Δϕ - lat_min,  ϕ_scale + Δϕ - lat_min], color=:white, linewidth=5)
text!(ax, λ_scale_l - lon_min, ϕ_scale + Δϕ - 0.000075 - lat_min, text = "30 m", color=:white, fontsize=12, font=:bold)

# add North arrow
scatter!(ax, [λ_scale_l - lon_min + 0.00003,], [ϕ_scale - lat_min + Δϕ + 0.00005], color=:white, marker=:utriangle, markersize=15)
text!(ax, [λ_scale_l - lon_min + 0.00001,], [ϕ_scale + Δϕ - lat_min + 0.000075], text="N", color=:white, fontsize=12, font=:bold)

cb = Colorbar(fig[1,2], colormap=cmap, colorrange=clims, minorticksvisible=true, label="Rhodamine Abundance")


# asp_rat = size(satmap.img, 1)/size(satmap.img,2)
asp_rat = (lon_max-lon_min)/(lat_max-lat_min)
colsize!(fig.layout, 1, Aspect(1, asp_rat))
resize_to_layout!(fig)

fig

save(joinpath(figpath, "plume-map-1.png"), fig; px_per_unit=2)




# plot second capture

lon_min, lon_max = (-97.7168, -97.7155)
lat_min, lat_max = (33.7015, 33.70275)

lon_l, lon_h = extrema(Lon2)
lat_l, lat_h = extrema(Lat2)

# set up color map
c_high = colorant"#f73a2d"
c_low = colorant"#450a06"
cmap = cgrad([c_low, c_high])
lim_low = 0.3
lim_high = 1.0
clims = (lim_low, lim_high)

area_pts_2 = [z for z ∈ Zpred2[1:330, 150:end] if !isnan(z) .&& z .> lim_low]
area_pts_3 = [z for z ∈ Zpred3[1:350, 1:200] if !isnan(z) .&& z .> lim_low]
tot_area = (length(area_pts_2) + length(area_pts_3)) * (0.1^2)

fig = Figure();
ax = CairoMakie.Axis(
    fig[1,1],
    xlabel="Longitude", xtickformat = x -> string.(round.(x .+ lon_min, digits=6)), xticklabelsize= 14,
    xticksize=7,
    xminorticksvisible=true,
    ylabel="Latitude",  ytickformat = y -> string.(round.(y .+ lat_min, digits=6)), yticklabelsize = 14,
    yticksize=7,
    yminorticksvisible=true,
    title="Total Area = $(round(tot_area, digits=1)) m²",
    titlealign=:left,
    titlefont=:regular,
    titlesize=13,
);
bg = heatmap!(
    ax,
    (satmap.w - lon_min)..(satmap.e - lon_min),
    (satmap.s - lat_min)..(satmap.n - lat_min),
    satmap.img,
    alpha=0.65,
)

h = heatmap!(ax, (lon_l-lon_min)..(lon_h-lon_min), (lat_l-lat_min)..(lat_h-lat_min), Zpred2, colormap=cmap, colorrange=clims, lowclip=:transparent)

lon_l, lon_h = extrema(Lon3)
lat_l, lat_h = extrema(Lat3)
h = heatmap!(ax, (lon_l-lon_min)..(lon_h-lon_min), (lat_l-lat_min)..(lat_h-lat_min), Zpred3, colormap=cmap, colorrange=clims, lowclip=:transparent)


Δϕ = 33.7015 - 33.70075
xlims!(ax, 0, lon_max - lon_min)
ylims!(ax, 0, lat_max - lat_min)

# add 30 meter scale bar
lines!(ax, [λ_scale_l - lon_min, λ_scale_r - lon_min], [ϕ_scale + Δϕ - lat_min,  ϕ_scale + Δϕ - lat_min], color=:white, linewidth=5)
text!(ax, λ_scale_l - lon_min, ϕ_scale + Δϕ - 0.000075 - lat_min, text = "30 m", color=:white, fontsize=12, font=:bold)

# add North arrow
scatter!(ax, [λ_scale_l - lon_min + 0.00003,], [ϕ_scale - lat_min + Δϕ + 0.00005], color=:white, marker=:utriangle, markersize=15)
text!(ax, [λ_scale_l - lon_min + 0.00001,], [ϕ_scale + Δϕ - lat_min + 0.000075], text="N", color=:white, fontsize=12, font=:bold)

cb = Colorbar(fig[1,2], colormap=cmap, colorrange=clims, minorticksvisible=true, label="Rhodamine Abundance")


# asp_rat = size(satmap.img, 1)/size(satmap.img,2)
asp_rat = (lon_max-lon_min)/(lat_max-lat_min)
colsize!(fig.layout, 1, Aspect(1, asp_rat))
resize_to_layout!(fig)

fig

save(joinpath(figpath, "plume-map-2.png"), fig; px_per_unit=2)



