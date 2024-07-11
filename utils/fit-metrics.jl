using LinearAlgebra
using Distances

# evaluate performance using spectral angle
function spectral_angle(r1, r2)
    return acosd(dot(r1, r2)/(norm(r1) * norm(r2)))  # return angle in degrees
end

function rmse(r1, r2)
    # return sqrt(sum((r1 .- r2).^2)/length(r1))
    return rmsd(r1, r2)
end

function spectral_information_divergence(r1, r2)
    # add 10 to deal with any negative values resulting from added noise.
    return kl_divergence(r1 .+ 10, r2 .+ 10) + kl_divergence(r2 .+ 10, r1 .+ 10)
end

