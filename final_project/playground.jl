using LinearAlgebra
using FileIO
using Statistics
using Plots
using Printf
include("helper.jl")

hyp_wl, rad = read_Hyperion()
profile, σ_matrix = make_vapor_xsection()
x = σ_matrix[:,1,1];
FWHM = 1.0;
ν_coarse = 1e4 ./ hyp_wl;
hyp = KernelInstrument(gaussian_kernel(FWHM, 0.002), ν_coarse);
σ_conv = conv_spectra(hyp, ν, x);
k1 = 4π * σ_conv ./ hyp_wl * 1e4;
k2, k3 = read_Kuo_abs(hyp_wl)

p = plot(hyp_wl, k1 * 1e19, yscale=:log10, dpi=400)
plot!(hyp_wl, k2)
plot!(hyp_wl, k3)
xlabel!("wavelength (um)")
ylabel!("absorption cross-section (cm-1)")
savefig(p, "cross_sections.png")

# m = minimum(rad, dims=2)[:,1,:];
# bad_indices = findall(x -> x < 0, m);
# println(size(bad_indices))
# println(bad_indices[1])



pts = [(5,10),(40,500),(80,1000),(120,1500),(160,2000),(200,2500),(240,3000)]

s = 20
rad_s = rad[:,s,:]'
p1 = heatmap(rad_s, clims=(0,3000), size=(400,500), title=@sprintf("radiance at %.3f (um)", hyp_wl[s]))
scatter!(pts, label="")
xlims!(0,256)
ylims!(0,3189)

p2 = plot()
for (i,j) in pts
    plot!(hyp_wl,rad[i,:,j], label="("*string(i)*", "*string(j)*")")
end
y = fwd_model([500, 10, 1e16, 1e-2, 1e-2])
plot!(hyp_wl,y)

xlims!(1.4,1.8)
ylims!(0,1000)
xlabel!("wavelength (um)")
ylabel!("radiance (W/m2/sr/um)")
plot(p1,p2,layout=(1,2),size=(800,500))