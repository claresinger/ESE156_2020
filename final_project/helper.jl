using DelimitedFiles
using NCDatasets
using Interpolations
using JLD2
using LinearAlgebra
using Plots
using Printf
include("../scripts/ese156_tools.jl")

function read_Hyperion()
    entity_id = "EO1H2221282005350110KF";
    file = "../EO1_Hyperion_data/"*entity_id*"/"*entity_id*".AUX";
    ds = Dataset(file);
    wl = ds["Spectral Center Wavelengths"][:];
    gain = ds["Gain Coefficients"][:];
    wl = mean(wl, dims=1)[1,:] * 1e-3;

    file = "../EO1_Hyperion_data/"*entity_id*"/"*entity_id*".L1R";
    ds = Dataset(file);
    ncross, nalong = ds.dim["fakeDim2"], ds.dim["fakeDim1"];
    nwl = ds.dim["fakeDim0"];
    rad = ds["EO1H2221282005350110KF.L1R"][:];

    wlrange = findall(x -> x > 1.4 && x < 1.8, wl);
    rad = rad[:,wlrange,:];
    gain = gain[:,wlrange];
    wl = wl[wlrange];
    
    return wl, rad, gain
end

function read_Kuo_abs(hyp_wl)
    dat = readdlm("Kuo93_imaginary_k.csv", ',', skipstart=1);
    wl_liq = reverse(dat[:,1]);
    k_liq = reverse(dat[:,2]);
    itp = LinearInterpolation(wl_liq, k_liq);
    interp_k_liq = itp(hyp_wl);
    k2 = 4π * interp_k_liq ./ hyp_wl * 1e4;

    k_ice = reverse(dat[:,3]);
    w = prepend!(findall(x -> x > 0, k_ice),[1]);
    wl_ice = wl_liq[w];
    k_ice = k_ice[w];
    k_ice[1] = 2e-5;
    itp = LinearInterpolation(wl_ice, k_ice);
    interp_k_ice = itp(hyp_wl);
    k3 = 4π * interp_k_ice ./ hyp_wl * 1e4;
    
    return k2, k3
end

function save_σ_hr()
    # Read profile (and generate dry/wet VCDs per layer)
    file = "../files/MERRA300.prod.assim.inst6_3d_ana_Nv.20150613.hdf.nc4"
    timeIndex = 3 # There is 00, 06, 12 and 18 in UTC, i.e. 6 hourly data stacked together
    myLat = -77.528 ;
    myLon = 167.164; 
    profile_hr = read_atmos_profile(file, myLat, myLon, timeIndex);

    # Minimum, maximum wavenumber
    ν_min, ν_max = 5540.0, 7160.0

    # read hitran cross-section, define hitran model
    h2o_par = CrossSection.read_hitran("../files/hitran_molec_id_1_H2O.par", mol=1, iso=1, ν_min=ν_min, ν_max=ν_max);
    h2o_voigt = make_hitran_model(h2o_par, Voigt(), wing_cutoff=10)

    # define matrix of cross sections
    res = 0.005
    ν = 5550:res:7160
    hitran_array = [h2o_voigt];
    σ_matrix_hr = compute_profile_crossSections(profile_hr, hitran_array , ν);
    
    @save "σ_hr.jld2" ν profile_hr σ_matrix_hr
end

function make_vapor_xsection()
    @load "σ_hr.jld2" ν profile_hr σ_matrix_hr
    n_layers = 20
    profile, σ_matrix = reduce_profile(n_layers, profile_hr, σ_matrix_hr);
    
    wv_wl = 1e4 ./ ν;
    σ_wv = σ_matrix[:,end,1];
    k1 = 4π * σ_wv ./ wv_wl * 1e4;
    
    return ν, k1
end

function plot_cross_sections(ν, k1, hyp_wl, k2, k3)
    ν_coarse = ν[1:100:end];
    hyp = KernelInstrument(gaussian_kernel(0.1, 0.005), ν_coarse);
    k1_conv = conv_spectra(hyp, ν, k1);
    coarse_wl = 1e4 ./ ν_coarse;

    p = plot(coarse_wl, k1_conv * 1e19, color=:blue, yscale=:log10, figsize=(400,400), label="water vapor")
    plot!(hyp_wl, k2, color=:green, lw=2, label="liquid")
    plot!(hyp_wl, k3, color=:red, lw=2, label="ice")
    xlabel!("wavelength (um)")
    ylabel!("absorption cross-section (cm-1)")
    savefig(p, "./figures/cross_sections.png")
    display(p)
end

function plot_radiances(hyp_wl, rad)
    pts = [(5,10),(40,500),(80,1000),(120,1500),(160,2000),(200,2500),(240,3000)]

    s = 20
    rad_s = rad[:,s,:]'
    p1 = heatmap(rad_s, clims=(0,1500), size=(400,500), title=@sprintf("radiance at %.3f (um)", hyp_wl[s]))
    scatter!(pts, color=1, label="")
    xlims!(0,256)
    ylims!(0,3189)

    p2 = plot()
    for (i,j) in pts
        plot!(hyp_wl,rad[i,:,j], label="("*string(i)*", "*string(j)*")")
    end

    xlims!(1.4,1.8)
    ylims!(0,1000)
    xlabel!("wavelength (um)")
    ylabel!("radiance (W/m2/sr/um)")
    p = plot(p1,p2,layout=(1,2),size=(800,500))
    savefig(p, "./figures/snapshot_data.png")
    display(p)
end

function read_solar()
    ds = readdlm("solar-visie.tsv",'\t',skipstart=35)
    wl_hr = ds[:,1];
    solar_hr = ds[:,2] .* 1e3;
    hyp = KernelInstrument(gaussian_kernel(0.001, 0.005), hyp_wl .* 1e3);
    Fsolar = conv_spectra(hyp, wl_hr[1]:wl_hr[2]-wl_hr[1]:wl_hr[end], solar_hr);
    wl_hr /= 1e3;

    p=plot(wl_hr, solar_hr);
    plot!(hyp_wl, Fsolar);
    xlabel!("wavelength (um)");
    ylabel!("solar irradiance (W/m2/um)");
    
    return Fsolar
end

function fwd_model(x; instrument=hyp, λ=hyp_wl, ν_hr=ν_hr, k1_hr=k1_hr, k2=k2, k3=k3)
    l, m, u1, u2, u3 = x;
    T0 = exp.(l .+ λ*m);
    T1_hr = exp.(-k1_hr*u1);
    T1_conv = conv_spectra(instrument, ν_hr, T1_hr);
    T23 = exp.(-k2*u2 .- k3*u3);
    ρ = T0 .* T1_conv .* T23;
    return ρ
end

function run_fit(y; λ=hyp_wl, k1=k1, k2=k2, k3=k3)
    K = [ones(length(λ)) λ -k1 -k2 -k3];
    
    # Construct error covariance matrix
    noise = 1/100;
    Se = Diagonal((ones(length(λ)).*noise).^2);
    
    # Solve normal equations:
    x̂ = inv(K'inv(Se)K) * K'inv(Se) * log.(y);
    
    return x̂
end

function chi_by_eye(hyp_wl, rad, gain, cosθ, Fsolar)
    (i,j) = (120,500);
    ρ = rad[i,:,j] .* gain[i,:] .* π ./ cosθ ./ Fsolar;
    p = plot(hyp_wl, ρ, color=:purple, label="("*string(i)*", "*string(j)*")");
    x̂ = [-1.4, 0.25, 5e15, 0.014, 0.009];
    plot!(hyp_wl, fwd_model(x̂), color=:purple, lw=2, label="mixed-phase", legend=:topleft);
    
    (i,j) = (240,3000);
    ρ = rad[i,:,j] .* gain[i,:] .* π ./ cosθ ./ Fsolar;
    plot!(hyp_wl, ρ, color=:green, label="("*string(i)*", "*string(j)*")");
    x̂ = [-1.9, 0.1, 7e15, 0.02, 0.004];
    plot!(hyp_wl, fwd_model(x̂), color=:green, lw=2, label="mostly liquid");

    (i,j) = (245,400);
    ρ = rad[i,:,j] .* gain[i,:] .* π ./ cosθ ./ Fsolar;
    p = plot!(hyp_wl, ρ, color=:red, label="("*string(i)*", "*string(j)*")");
    x̂ = [-2.3, 0.7, 5e15, 0.002, 0.014];
    plot!(hyp_wl, fwd_model(x̂), color=:red, lw=2, label="mostly ice");
    
    xlims!(1.4,1.8)
    xlabel!("wavelength (um)")
    ylabel!("TOA reflectance")
    savefig(p, "./figures/chi-by-eye.png")
    display(p)
end

function rho_fit(hyp_wl, rad, gain, cosθ, Fsolar)
    (i,j) = (40,500);
    y = rad[i,:,j] .* gain[i,:] .* π ./ cosθ ./ Fsolar;
    x̂ = run_fit(y);
    LTF = x̂[4] / (x̂[4] + x̂[5]);

    p1 = plot(hyp_wl, y, label="measured", legend=:bottomright);
    plot!(hyp_wl, fwd_model(x̂), label="modeled");
    ylabel!("TOA reflectance");
    ylims!(0,0.3)
    title!(@sprintf("LTF=%.2f",LTF))

    instrument = KernelInstrument(gaussian_kernel(10, 0.005), 1e4 ./ hyp_wl);
    Tvapor_hr = exp.(-k1_hr*x̂[3]);
    Tvapor_conv = conv_spectra(instrument, ν_hr, Tvapor_hr);
    p3 = plot(hyp_wl, Tvapor_conv, color=:blue, label="water vapor", legend=:bottomright);
    plot!(hyp_wl, exp.(-x̂[4]*k2), color=:green, label="liquid");
    plot!(hyp_wl, exp.(-x̂[5]*k3), color=:red, label="ice");
    ylabel!("transmittance")
    ylims!(0,1)

    p5 = plot(hyp_wl, fwd_model(x̂) - y, color=2, label="");
    plot!(hyp_wl, zeros(length(hyp_wl)), color=:black, label="");
    xlabel!("wavelength (um)");
    ylabel!("residual");
    ylims!(-0.1,0.1)

    (i,j) = (240,3000);
    y = rad[i,:,j] .* gain[i,:] .* π ./ cosθ ./ Fsolar;
    x̂ = run_fit(y);
    LTF = x̂[4] / (x̂[4] + x̂[5]);

    p2 = plot(hyp_wl, y, label="");
    plot!(hyp_wl, fwd_model(x̂), label="");
    ylims!(0,0.3)
    title!(@sprintf("LTF=%.2f",LTF))

    p4 = plot(hyp_wl, exp.(-x̂[3]*k1), color=:blue, label="");
    plot!(hyp_wl, exp.(-x̂[4]*k2), color=:green, label="");
    plot!(hyp_wl, exp.(-x̂[5]*k3), color=:red, label="");
    ylims!(0,1)

    p6 = plot(hyp_wl, fwd_model(x̂) - y, color=2, label="");
    plot!(hyp_wl, zeros(length(hyp_wl)), color=:black, label="");
    xlabel!("wavelength (um)");
    ylims!(-0.1,0.1)

    p = plot(p1,p2,p3,p4,p5,p6,layout=(3,2),size=(700,500))
    savefig(p, "./figures/rho_transmittance_residual.png")
    display(p)
end