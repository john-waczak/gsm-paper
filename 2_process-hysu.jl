using CondaPkg
CondaPkg.add("numpy")
CondaPkg.add("tifffile")
using PythonCall
using HDF5
using MAT
using Images, ImageIO
import CairoMakie as cmk
using DataFrames, CSV

tifffile = pyimport("tifffile")

path_large_targets = "./data/hysu/original/HySpex/large_targets"
path_small_targets = "./data/hysu/original/HySpex/small_targets"
path_all_targets = "./data/hysu/original/HySpex/all_targets"
path_full = "./data/hysu/original/HySpex/full"

masks_path = joinpath(path_all_targets, "TargetsROIs", "Matlab", "masks.mat")

# all pixels in all_targets that include on the 6 classes
pixel_mask = .!(Bool.(load(joinpath(path_all_targets, "DLR_HySU_HS_all_targets_mask.png"))))

size(pixel_mask)

# load data from tiff files
X_large = tifffile.tifffile.imread(joinpath(path_large_targets, "DLR_HySU_HS_large_targets.tif"))
X_small = tifffile.tifffile.imread(joinpath(path_small_targets, "DLR_HySU_HS_small_targets.tif"))
X_all= tifffile.tifffile.imread(joinpath(path_all_targets, "DLR_HySU_HS_all_targets.tif"))
X_full = tifffile.tifffile.imread(joinpath(path_full, "DLR_HySU_HS_full.tif"))

# convert to Julia Array
X_large = pyconvert(Array, X_large);
X_small = pyconvert(Array, X_small);
X_all = pyconvert(Array, X_all);
X_full = pyconvert(Array, X_full);

# rescale to reflectance ∈ [0,1]
scale_fac = 10_000.0

X_large = X_large/scale_fac ;
X_small = X_small/scale_fac ;
X_all = X_all/scale_fac ;
X_full = X_full/scale_fac ;


# load in area masks for "all_targets"
masks = matread(masks_path)
masks

size(masks["materials"])
size(masks["targetsROI"])


size(X_large)
size(X_small)
size(X_all)

masks["materials"];

fig0 = cmk.heatmap(masks["materials"][:,:,1])
fig1 = cmk.heatmap(masks["targetsROI"])
fig2 = cmk.heatmap(X_full[:,:,100])

wavelengths = [0.417400, 0.421020, 0.424640, 0.428270, 0.431890, 0.435510, 0.439130, 0.442760, 0.446380, 0.450000, 0.453620, 0.457250, 0.460870, 0.464490, 0.468110, 0.471730, 0.475360, 0.478980, 0.482600, 0.486220, 0.489850, 0.493470, 0.497090, 0.500710, 0.504340, 0.507960, 0.511580, 0.515200, 0.518830, 0.522450, 0.526070, 0.529690, 0.533310, 0.536940, 0.540560, 0.544180, 0.547800, 0.551430, 0.555050, 0.558670, 0.562290, 0.565920, 0.569540, 0.573160, 0.576780, 0.580400, 0.584030, 0.587650, 0.591270, 0.594890, 0.598520, 0.602140, 0.605760, 0.609380, 0.613010, 0.616630, 0.620250, 0.623870, 0.627490, 0.631120, 0.634740, 0.638360, 0.641980, 0.645610, 0.649230, 0.652850, 0.656470, 0.660100, 0.663720, 0.667340, 0.670960, 0.674590, 0.678210, 0.681830, 0.685450, 0.689070, 0.692700, 0.696320, 0.699940, 0.703560, 0.707190, 0.710810, 0.714430, 0.718050, 0.721680, 0.725300, 0.728920, 0.732540, 0.736160, 0.739790, 0.743410, 0.747030, 0.750650, 0.754280, 0.757900, 0.761520, 0.765140, 0.768770, 0.772390, 0.776010, 0.779630, 0.783260, 0.786880, 0.790500, 0.794120, 0.797740, 0.801370, 0.804990, 0.808610, 0.812230, 0.815860, 0.819480, 0.823100, 0.826720, 0.830350, 0.833970, 0.837590, 0.841210, 0.844830, 0.848460, 0.852080, 0.855700, 0.859320, 0.862950, 0.866570, 0.870190, 0.873810, 0.877440, 0.881060, 0.884680, 0.888300, 0.891920, 0.895550, 0.899170, 0.902790]

fwhm = [ 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004870, 0.004850, 0.004820, 0.004800, 0.004800, 0.004800, 0.004830, 0.004850, 0.004870,  0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004870, 0.004840, 0.004820, 0.004800, 0.004800, 0.004810, 0.004830, 0.004850, 0.004870, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004860, 0.004840, 0.004820, 0.004800, 0.004800, 0.004810, 0.004830, 0.004850, 0.004870, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004860, 0.004840, 0.004820, 0.004800, 0.004800, 0.004810, 0.004830, 0.004850, 0.004880, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004880, 0.004860, 0.004840, 0.004820, 0.004800, 0.004800, 0.004810, 0.004830, 0.004860, 0.004880, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004880, 0.004860, 0.004840, 0.004810, 0.004800, 0.004800, 0.004810, 0.004840, 0.004860, 0.004880, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004880, 0.004860, 0.004830, 0.004810, 0.004800, 0.004800, 0.004810, 0.004840, 0.004860, 0.004880, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004760, 0.004880, 0.004860, 0.004830, 0.004810, 0.004800]




data_out = h5open("./data/hysu/processed/data.h5", "w")

data_out["large"] = X_large;
data_out["small"] = X_small;
data_out["all"] = X_all;
data_out["full"] = X_full;
data_out["materials"] = masks["materials"];
data_out["targetsROI"] = masks["targetsROI"];
data_out["pixel_mask"] = Int8.(pixel_mask);
data_out["wavelengths"] = wavelengths;
data_out["fwhm"] = fwhm;

close(data_out)



# create dataframe from the "all", "large", and "small" datasets

λs = ["λ_"*lpad(i, 3, "0") for i ∈ 1:length(wavelengths)]

idxs = findall(pixel_mask)
X_all[:, idxs]

df_all = DataFrame(X_all[:, idxs]', λs);
CSV.write("./data/hysu/all.csv", df_all)

df_large = DataFrame(X_large[:,[CartesianIndex(i,j) for i ∈ axes(X_large,2) for j ∈ axes(X_large,3)]]', λs);
CSV.write("./data/hysu/large.csv", df_large)

df_small= DataFrame(X_small[:,[CartesianIndex(i,j) for i ∈ axes(X_small,2) for j ∈ axes(X_small,3)]]', λs);
CSV.write("./data/hysu/small.csv", df_large)

