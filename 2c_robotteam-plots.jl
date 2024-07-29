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


include("./utils/fit-metrics.jl")


figpath = "./figures/robot-team"
outpath = "./output/robot-team"
json_path = joinpath(outpath, "fit_final.json")
mdl_path = joinpath(outpath, "model_final.jls")

res_dict = JSON.parsefile(json_path)
mach = machine(mdl_path)
rpt = report(mach)

Nv = res_dict["Nv"]
W = rpt[:W]

df_features = CSV.read("./data/robot-team/df_features.csv", DataFrame);

idx_vertices =rpt[:idx_vertices]
node_means =rpt[:node_data_means]
Rorig = Array(df_features);

Ypred = data_reconstruction(mach, df_features)
Zpred = DataFrame(MLJ.transform(mach, df_features))
idx1_max = argmax(Zpred.Z1)
idx2_max = argmax(Zpred.Z2)
idx3_max = argmax(Zpred.Z3)
# visualize endmembers

# c_water = mints_colors[3]
# c_veg = mints_colors[1]
# c_dye = mints_colors[2]

c_water = CairoMakie.RGBf(0,0,1)
c_veg = CairoMakie.RGBf(0,1,0)
c_dye = CairoMakie.RGBf(1,0,0)

fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="λ (nm)", ylabel="Reflectance")

# l_veg = lines!(ax, wavelengths, node_means[:,idx_vertices[1]], color=c_veg)
# l_water = lines!(ax, wavelengths, node_means[:,idx_vertices[2]], color=c_water)
# l_dye = lines!(ax, wavelengths, node_means[:,idx_vertices[3]], color=c_dye)

l_veg = lines!(ax, wavelengths, Ypred[idx1_max,:], color=c_veg)
l_water = lines!(ax, wavelengths, Ypred[idx2_max,:], color=c_water)
l_dye = lines!(ax, wavelengths, Ypred[idx3_max,:], color=c_dye)

fig[1,1] = Legend(fig, [l_water, l_veg, l_dye], ["Water", "Vegetation", "Rhodamine"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=14, height=-5)

xlims!(ax, wavelengths[1], wavelengths[end])
ylims!(ax, 0, nothing)

fig

save(joinpath(figpath, "extracted-endmembers.png"), fig)
save(joinpath(figpath, "extracted-endmembers.pdf"), fig)




# load the model
datapath = "./data/robot-team"
exemplar_cube_path = joinpath(datapath, "sample-datacube.h5")
@assert ispath(exemplar_cube_path)


# generate rgb image for background
rgb_image = get_h5_rgb(exemplar_cube_path)


# open HDF5 file and gather data
h5 = h5open(exemplar_cube_path, "r")
Δx = 0.1
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
idx_900 = findfirst(wavelengths .≥ 900)
Data[idx_NDWI, :, :]

IsPlume = zeros(Bool, size(IsInbounds))
IsPlume[1:250, 150:end] .= true

ij_nw = findall(IsInbounds .&& (Data[idx_NDWI,:,:] .> 0.25))
ij_plume = findall(IsPlume .&& IsInbounds)

df_plume = DataFrame(Data[1:idx_900,ij_plume]', varnames[1:idx_900])
df_nw = DataFrame(Data[1:idx_900,ij_nw]', varnames[1:idx_900])


# set up output array for Zs
Zpred = fill(NaN, size(Data)[2:end]..., 3);
Zdom = fill(4, size(Data)[2:end]...);

# create dataframe with inferred abundances
df_z_plume = DataFrame(MLJ.transform(mach, df_plume))
df_z_nw = DataFrame(MLJ.transform(mach, df_nw))
dom_end_plume = [argmax(Array(row)) for row ∈ eachrow(df_z_plume)]
dom_end_nw = [argmax(Array(row)) for row ∈ eachrow(df_z_nw)]

for i ∈ 1:Nv
    Zpred[ij_nw, i] .= df_z_nw[:,i]
    Zpred[ij_plume, i] .= df_z_plume[:,i]
end
Zdom[ij_nw] .= dom_end_nw
Zdom[ij_plume] = dom_end_plume

# create classification map
c_empty = CairoMakie.RGBAf(0,0,0,0)
clist = [c_veg, c_water, c_dye, c_empty]

# Fill in classification map
Z_pred_color = zeros(CairoMakie.RGBAf, size(X))
Z_dom_color = zeros(CairoMakie.RGBAf, size(X))

for j ∈ axes(Z_pred_color,2), i ∈ axes(Z_pred_color, 1)
    if isnan(Zpred[i,j,1])
        Z_pred_color[i,j] = c_empty
    else
        Z_pred_color[i,j] = Zpred[i,j,1]*c_veg + Zpred[i,j,2]*c_water + Zpred[i,j,3]*c_dye
    end
end

for i ∈ 1:length(Z_dom_color)
    Z_dom_color[i] = clist[Zdom[i]]
end



lon_min, lon_max = extrema(Longitudes)
lat_min, lat_max = extrema(Latitudes)


# generate scale bar and north arrow
ϕ_scale = 33.701860
m_per_deg = 111412.84*cosd(ϕ_scale) - 93.5*cosd(3*ϕ_scale) + 0.118 * cosd(5*ϕ_scale)
λ_scale_l = -97.71565
λ_scale_r = λ_scale_l + 10/m_per_deg


# visualize class map
fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="Longitude", ylabel="Latitude", xticklabelsize=15, yticklabelsize=15);
image!(ax, lon_min..lon_max, lat_min..lat_max, rgb_image, alpha=0.25)
image!(ax, lon_min..lon_max, lat_min..lat_max, Z_pred_color)

fig[1,1] = Legend(fig, [l_water, l_veg, l_dye], ["Water", "Vegetation", "Rhodamine"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=14, height=-5)

asp_rat = size(rgb_image, 2)/size(rgb_image,1)
rowsize!(fig.layout, 2, Aspect(1, asp_rat))
resize_to_layout!(fig)

# add 30 meter scale bar
lines!(ax, [λ_scale_l, λ_scale_r], [ϕ_scale, ϕ_scale], color=colorant"#898989", linewidth=3)
text!(ax, 0.75*(λ_scale_l) + 0.25*(λ_scale_r), ϕ_scale - 0.00002, text = "10 m", color=colorant"#898989", fontsize=12, font=:regular)

# add North arrow
scatter!(ax, [λ_scale_r + 0.00003,], [ϕ_scale,], color=colorant"#898989", marker=:utriangle, markersize=25)
text!(ax, [λ_scale_r + 0.0000225,], [ϕ_scale + 0.000015], text="N", color=colorant"#898989", fontsize=12, font=:bold)
fig

save(joinpath(figpath, "datacube-gsm-colormap.png"), fig; px_per_unit=2)


fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="Longitude", ylabel="Latitude", xticklabelsize=15, yticklabelsize=15);
image!(ax, lon_min..lon_max, lat_min..lat_max, rgb_image, alpha=0.35)
image!(ax, lon_min..lon_max, lat_min..lat_max, Z_dom_color)

fig[1,1] = Legend(fig, [l_water, l_veg, l_dye], ["Water", "Vegetation", "Rhodamine"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=14, height=-5)

asp_rat = size(rgb_image, 2)/size(rgb_image,1)
rowsize!(fig.layout, 2, Aspect(1, asp_rat))
resize_to_layout!(fig)

# add 30 meter scale bar
lines!(ax, [λ_scale_l, λ_scale_r], [ϕ_scale, ϕ_scale], color=colorant"#898989", linewidth=3)
text!(ax, 0.75*(λ_scale_l) + 0.25*(λ_scale_r), ϕ_scale - 0.00002, text = "10 m", color=colorant"#898989", fontsize=12, font=:regular)

# add North arrow
scatter!(ax, [λ_scale_r + 0.00003,], [ϕ_scale,], color=colorant"#898989", marker=:utriangle, markersize=25)
text!(ax, [λ_scale_r + 0.0000225,], [ϕ_scale + 0.000015], text="N", color=colorant"#898989", fontsize=12, font=:bold)
fig

save(joinpath(figpath, "datacube-gsm-colormap-dominant.png"), fig; px_per_unit=2)


# visualize the rgb image
fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="Longitude", ylabel="Latitude", xticklabelsize=15, yticklabelsize=15);
image!(ax, lon_min..lon_max, lat_min..lat_max, rgb_image)
asp_rat = size(rgb_image, 1)/size(rgb_image,2)
colsize!(fig.layout, 1, Aspect(1, asp_rat))
resize_to_layout!(fig)

# add 30 meter scale bar
lines!(ax, [λ_scale_l, λ_scale_r], [ϕ_scale, ϕ_scale], color=colorant"#898989", linewidth=3)
text!(ax, 0.75*(λ_scale_l) + 0.25*(λ_scale_r), ϕ_scale - 0.00002, text = "10 m", color=colorant"#898989", fontsize=12, font=:regular)

# add North arrow
scatter!(ax, [λ_scale_r + 0.00003,], [ϕ_scale,], color=colorant"#898989", marker=:utriangle, markersize=25)
text!(ax, [λ_scale_r + 0.0000225,], [ϕ_scale + 0.000015], text="N", color=colorant"#898989", fontsize=12, font=:bold)
fig

save(joinpath(figpath, "datacube-rgb.png"), fig, px_per_unit=2)




# now let's do maps for each endmember alone...
c_high = colorant"#f73a2d"
c_low = colorant"#450a06"
cmap = cgrad([c_low, c_high])

extrema([z for z in Zpred[:,:,3] if !isnan(z)])
q_low = quantile([z for z in Zpred[:,:,3] if !isnan(z)], 0.75)
q_high = quantile([z for z in Zpred[:,:,3] if !isnan(z)], 0.99)

lim_low = 0.3
lim_high = 1.0
clims = (lim_low, lim_high)

fig = Figure();
ax = CairoMakie.Axis(fig[1,1],
          xlabel="Longitude", ylabel="Latitude",
          xticklabelsize=15, yticklabelsize=15,
          );
image!(ax, lon_min..lon_max, lat_min..lat_max, rgb_image, alpha=0.65)

llon_min, llon_max = extrema(Longitudes[1:250, 150:end])
llat_min, llat_max = extrema(Latitudes[1:250, 150:end])
heatmap!(ax, llon_min..llon_max, llat_min..llat_max, Zpred[1:250, 150:end,3], colormap=cmap, colorrange=clims, lowclip=:transparent)

cb = Colorbar(fig[1,2],
              colormap=cmap, colorrange=clims,
              ticklabelsize=15,
              label="Rhodamine Abundance",
              labelsize=16,
              minorticksvisible=true
              )


area_pts = [z for z ∈ Zpred[1:250, 150:end, 3] if !isnan(z) .&& z .> lim_low]
tot_area = length(area_pts) * (0.1^2)

text!(fig.scene, 0.610, 0.93, text="Total Area = $(round(tot_area, digits=1)) m²", space=:relative, fontsize=15)

asp_rat = size(rgb_image, 1)/size(rgb_image,2)
colsize!(fig.layout, 1, Aspect(1, asp_rat))
resize_to_layout!(fig)

# add 30 meter scale bar
lines!(ax, [λ_scale_l, λ_scale_r], [ϕ_scale, ϕ_scale], color=colorant"#898989", linewidth=3)
text!(ax, 0.75*(λ_scale_l) + 0.25*(λ_scale_r), ϕ_scale - 0.00002, text = "10 m", color=colorant"#898989", fontsize=12, font=:regular)

# add North arrow
scatter!(ax, [λ_scale_r + 0.00003,], [ϕ_scale,], color=colorant"#898989", marker=:utriangle, markersize=25)
text!(ax, [λ_scale_r + 0.0000225,], [ϕ_scale + 0.000015], text="N", color=colorant"#898989", fontsize=12, font=:bold)
fig

save(joinpath(figpath, "rhodmaine-plume.png"), fig; px_per_unit=2)



# vertex 2 map...
# c_high = colorant"#526cfa"
# c_low = colorant"#18204f"

c_high = colorant"#34eb49"
c_low = colorant"#07210a"
cmap = cgrad([c_low, c_high])

extrema([z for z in Zpred[:,:,2] if !isnan(z)])
q_low = quantile([z for z in Zpred[:,:,1] if !isnan(z)], 0.75)
q_high = quantile([z for z in Zpred[:,:,1] if !isnan(z)], 0.95)

clims = (q_low, q_high)

fig = Figure();
ax = CairoMakie.Axis(fig[1,1],
          xlabel="Longitude", ylabel="Latitude",
          xticklabelsize=15, yticklabelsize=15,
          );
image!(ax, lon_min..lon_max, lat_min..lat_max, rgb_image, alpha=0.65)
heatmap!(ax, lon_min..lon_max, lat_min..lat_max, Zpred[:, :,1], colormap=cmap, colorrange=clims, lowclip=:transparent, highclip=c_high)

cb = Colorbar(fig[1,2],
              colormap=cmap, colorrange=clims,
              ticklabelsize=15,
              label="Vegetation",
              labelsize=16,
              minorticksvisible=true
              )

area_pts = [z for z ∈ Zpred[:, :, 1] if !isnan(z) .&& z .≥ q_low]
tot_area = length(area_pts) * (0.1^2)

text!(fig.scene, 0.610, 0.93, text="Total Area = $(round(tot_area, digits=1)) m²", space=:relative, fontsize=15)

asp_rat = size(rgb_image, 1)/size(rgb_image,2)
colsize!(fig.layout, 1, Aspect(1, asp_rat))
resize_to_layout!(fig)

# add 30 meter scale bar
lines!(ax, [λ_scale_l, λ_scale_r], [ϕ_scale, ϕ_scale], color=colorant"#898989", linewidth=3)
text!(ax, 0.75*(λ_scale_l) + 0.25*(λ_scale_r), ϕ_scale - 0.00002, text = "10 m", color=colorant"#898989", fontsize=12, font=:regular)

# add North arrow
scatter!(ax, [λ_scale_r + 0.00003,], [ϕ_scale,], color=colorant"#898989", marker=:utriangle, markersize=25)
text!(ax, [λ_scale_r + 0.0000225,], [ϕ_scale + 0.000015], text="N", color=colorant"#898989", fontsize=12, font=:bold)
fig

save(joinpath(figpath, "vegetation.png"), fig; px_per_unit=2)



# vertex 1 map...
# c_high = colorant"#13fc03"
# c_low = colorant"#052603"
c_high = colorant"#2fd1fa"
c_low = colorant"#08242b"
cmap = cgrad([c_low, c_high])

extrema([z for z in Zpred[:,:,1] if !isnan(z)])
q_low = quantile([z for z in Zpred[:,:,2] if !isnan(z)], 0.25)
q_high = quantile([z for z in Zpred[:,:,2] if !isnan(z)], 0.9)

clims = (q_low, q_high)

fig = Figure();
ax = CairoMakie.Axis(fig[1,1],
          xlabel="Longitude", ylabel="Latitude",
          xticklabelsize=15, yticklabelsize=15,
          );
image!(ax, lon_min..lon_max, lat_min..lat_max, rgb_image, alpha=0.65)
heatmap!(ax, lon_min..lon_max, lat_min..lat_max, Zpred[:, :,2], colormap=cmap, colorrange=clims, lowclip=:transparent, highclip=c_high)

cb = Colorbar(fig[1,2],
              colormap=cmap, colorrange=clims,
              ticklabelsize=15,
              label="Water",
              labelsize=16,
              minorticksvisible=true,
              )

asp_rat = size(rgb_image, 1)/size(rgb_image,2)
colsize!(fig.layout, 1, Aspect(1, asp_rat))
resize_to_layout!(fig)

# add 30 meter scale bar
lines!(ax, [λ_scale_l, λ_scale_r], [ϕ_scale, ϕ_scale], color=colorant"#898989", linewidth=3)
text!(ax, 0.75*(λ_scale_l) + 0.25*(λ_scale_r), ϕ_scale - 0.00002, text = "10 m", color=colorant"#898989", fontsize=12, font=:regular)

# add North arrow
scatter!(ax, [λ_scale_r + 0.00003,], [ϕ_scale,], color=colorant"#898989", marker=:utriangle, markersize=25)
text!(ax, [λ_scale_r + 0.0000225,], [ϕ_scale + 0.000015], text="N", color=colorant"#898989", fontsize=12, font=:bold)
fig

save(joinpath(figpath, "water.png"), fig; px_per_unit=2)

fig
