using Images


function getRGB(h5::HDF5.File; λred=630.0, λgreen=532.0, λblue=465.0, Δx=0.10, α=10.0, β=0.0)
    λs = h5["data-Δx_$(Δx)/λs"][:]

    λred=630.0
    λgreen=532.0
    λblue=465.0

    idx_r = argmin(abs.(λs .- λred))
    idx_g = argmin(abs.(λs .- λgreen))
    idx_b = argmin(abs.(λs .- λblue))

    Rr = h5["data-Δx_$(Δx)/Data"][idx_r, :, :]
    Rg = h5["data-Δx_$(Δx)/Data"][idx_g, :, :]
    Rb = h5["data-Δx_$(Δx)/Data"][idx_b, :, :]

    ij_pixels = findall(h5["data-Δx_$(Δx)/IsInbounds"][:,:])
    img = zeros(4, size(Rr)...)

    Threads.@threads for ij ∈ ij_pixels
        img[1, ij] = Rr[ij]
        img[2, ij] = Rg[ij]
        img[3, ij] = Rb[ij]
        img[4, ij] = 1.0
    end

    return img
end


function gamma_correct(img, γ=1/2)
    # see https://en.wikipedia.org/wiki/Gamma_correction
    img_out = copy(img)

    img_out[1:3,:,:] .= img_out[1:3,:,:] .^ γ

    return img_out
end

function brighten(img, α=0.13, β=0.0)
    # see : https://www.satmapper.hu/en/rgb-images/
    img_out = copy(img)
    for i ∈ 1:3
        img_out[i,:,:] .= clamp.(α .* img_out[i,:,:] .+ β, 0, 1)
    end

    return img_out
end

function get_h5_rgb(h5path)
    h5 = h5open(h5path)
    img = getRGB(h5)
    close(h5)
    img_out = brighten(gamma_correct(img, .75), 3.5, 0)
    return colorview(RGBA, img_out)
end

