using DelimitedFiles
using NCDatasets
using Interpolations
using JLD2
include("../scripts/ese156_tools.jl")

function read_Hyperion()
    entity_id = "EO1H2221282005350110KF";
    file = "../EO1_Hyperion_data/"*entity_id*"/"*entity_id*".AUX";
    ds = Dataset(file);
    wl = ds["Spectral Center Wavelengths"][:];
    wl = mean(wl, dims=1)[1,:] * 1e-3;

    file = "../EO1_Hyperion_data/"*entity_id*"/"*entity_id*".L1R";
    ds = Dataset(file);
    ncross, nalong = ds.dim["fakeDim2"], ds.dim["fakeDim1"];
    nwl = ds.dim["fakeDim0"];
    rad = ds["EO1H2221282005350110KF.L1R"][:];

    wlrange = findall(x -> x > 1.4 && x < 1.8, wl);
    rad = rad[:,wlrange,:];
    wl = wl[wlrange];
    
    return wl, rad
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
    savefig(p, "cross_sections.png")
    display(p)
end