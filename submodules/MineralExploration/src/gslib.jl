@with_kw struct GSLIBDistribution <: GeoDist
    grid_dims::Tuple{Int64, Int64, Int64} = (50, 50, 1)
    n::Tuple{Int64, Int64, Int64} = (50, 50, 1) # same as grid_dims, renamed for convenience
    data::RockObservations = RockObservations()
    mean::Float64 = 0.3
    sill::Float64 = 0.005 # TODO currently defined by the target_histogram_file. Not used.
    nugget::Float64 = 0.0001
    variogram::Tuple = (1, 1, 0.0, 0.0, 0.0, 30.0, 30.0, 1.0)
    # # CHANGE RARELY
    target_histogram_file::String = "parameters/example_porosity.txt"
    columns_for_vr_and_wt = (1,0)
    zmin_zmax = (0.0, 1.0)
    lower_tail_option = (1, 0.0)
    upper_tail_option = (1, 1.0)

    # DO NOT CHANGE BELOW PARAMS
    transform_data::Bool = true
    mn = (0.5, 0.5, 0.5)
    sz = (1,1,1)
end

function write_string_to_file(fn, str)
    open(fn, "w") do io
        write(io, str)
    end
    fn
end

function data_to_string(data::RockObservations)
    str = """
    data
    4
    x
    y
    z
    poro
    """
    #TODO: Deal with cell-centered offsets?
    for i=1:length(data)
        str = string(str, data.coordinates[1,i] - 1, " ", data.coordinates[2,i] - 1, " 0.5 ", data.ore_quals[i], "\n")
    end
    str
end

function data_to_string_2d(data::RockObservations)
    str = """
    data
    3
    x
    y
    poro
    """
    #TODO: Deal with cell-centered offsets?
    for i=1:length(data)
        str = string(str, data.coordinates[1,i] - 1, " ", data.coordinates[2,i] - 1, " ", data.ore_quals[i], "\n")
    end
    str
end

function sgsim_params_to_string(p::GSLIBDistribution, data_file, N, dir, seed=nothing)
    if seed == nothing
        seed = rand(1:10000000)
    end
    """
    Parameters for SGSIM
********************

START OF PARAMETERS:
$(data_file)          -file with data
1  2  3  4  0  0              -  columns for X,Y,Z,vr,wt,sec.var.
-9999999 999999               -  trimming limits
1                             -transform the data (0=no, 1=yes)
$(dir)sgsim.trn                     -  file for output trans table
1                             -  consider ref. dist (0=no, 1=yes)
$(p.target_histogram_file)                  -  file with ref. dist distribution
$(p.columns_for_vr_and_wt[1])  $(p.columns_for_vr_and_wt[2])                          -  columns for vr and wt
$(p.zmin_zmax[1])    $(p.zmin_zmax[2])                     -  zmin,zmax(tail extrapolation)
$(p.lower_tail_option[1])       $(p.lower_tail_option[2])                     -  lower tail option, parameter
$(p.upper_tail_option[1])      $(p.upper_tail_option[2])                     -  upper tail option, parameter
1                             -debugging level: 0,1,2,3
$(dir)sgsim.dbg                     -file for debugging output
$(dir)sgsim.out                     -file for simulation output
$N                             -number of realizations to generate
$(p.n[1])    $(p.mn[1])    $(p.sz[1])              -nx,xmn,xsiz
$(p.n[2])    $(p.mn[2])    $(p.sz[1])              -ny,ymn,ysiz
$(p.n[3])    $(p.mn[3])    $(p.sz[1])              -nz,zmn,zsiz
$seed                         -random number seed
0     8                       -min and max original data for sim
12                            -number of simulated nodes to use
1                             -assign data to nodes(0=no, 1=yes)
1     3                       -multiple grid search(0=no,
0                             -maximum data per octant (0=not
100.0  100.0  10.0              -maximum search radii
0.0   0.0   0.0              -angles for search ellipsoid
51    51    11                -size of covariance lookup table
0     $(p.mean)   1.0              -ktype: 0=SK,1=OK,2=LVM,3=EXDR,
../data/ydata.dat             -  file with LVM, EXDR, or COLC
4                             -  column for secondary variable
1    $(0.02)                      -nst, nugget effect
$(p.variogram[1])    $(0.98)  $(p.variogram[3])   $(p.variogram[4])   $(p.variogram[5])     -it,cc,ang1,ang2,ang3
$(p.variogram[6])  $(p.variogram[7])  $(p.variogram[8])     -a_hmax, a_hmin, a_vert
"""
end

function kb2d_params_to_string(p::GSLIBDistribution, data_file, dir)
    """
Parameters for KB2D
********************

START OF PARAMETERS:
$data_file          -file with data
1  2  3                          -columns for X,Y,KV
-9999999 999999                  -trimming limits
1                                -debugging level: 0,1,2,3
$(dir)kb2d.dbg            -file for debugging output
$(dir)kb2d.out            -file for simulation output
$(p.n[1])    $(p.mn[1])    $(p.sz[1])                 -nx,xmn,xsiz
$(p.n[2])    $(p.mn[2])    $(p.sz[1])               -ny,ymn,ysiz
1   1                         -nxdis, nydis
0     8                       -min and max original data for sim
1000.0                         -radius
0     0.60                 -ktype: 0=SK,1=OK,2=LVM,3=EXDR,
$(p.nugget[1])    $(p.nugget[2])                     -nst, nugget effect
$(p.variogram[1])    $(p.variogram[2])  $(p.variogram[3]) $(p.variogram[6])  $(p.variogram[7])      -it,cc,ang1,a_hmax, a_hmin
"""
end

function write_sgsim_params_to_file(p::GSLIBDistribution, N; dir="./", out_fn="sgsim.par", data_fn="data.txt")
    data_file = write_string_to_file(string(dir, data_fn), data_to_string(p.data))
    write_string_to_file(string(dir, out_fn), sgsim_params_to_string(p, data_file, N, dir))
end

function write_kb2d_params_to_file(p::GSLIBDistribution, obs::RockObservations; dir="./", out_fn="kb2d.par", data_fn="data.txt")
    data_file = write_string_to_file(string(dir, data_fn), data_to_string_2d(obs))
    write_string_to_file(string(dir, out_fn), kb2d_params_to_string(p, data_file, dir))
end

function Base.rand(p::GSLIBDistribution, n::Int64=1, dir="sgsim_output/"; silent::Bool=true)
    # Write the param file
    if silent
        stdout_orig = stdout
        (rd, wr) = redirect_stdout()
    end
    errored = false
    try
        global ore_quals
        fn = write_sgsim_params_to_file(p, n; dir=dir) # NOTE: If we are going to want to sample many instances then we can include an "N" parameter here instead of the 1, but would need to update the code below as well

        # Run sgsim
        run(`sgsim $fn`)

    catch
        errored = true
    end
    if silent
        redirect_stdout(stdout_orig)
        close(rd)
        close(wr)
    end
    if errored
        error("SGSIM sampling error!")
    end
    # Load the results and return
    vals = CSV.File("$(dir)sgsim.out",header=3, types=Float64) |> CSV.Tables.matrix
    # reshape(vals, p.n..., N) # For multiple samples

    if n==1
        poro_2D = reshape(vals, p.n)
        ore_quals = repeat(poro_2D, outer=(1, 1, p.grid_dims[3]))
        return ore_quals
    else
        ore_quals = Array{Float64, 3}[]
        increment = prod(p.grid_dims)
        for i = 1:n
            start_idx = (i - 1)*increment + 1
            end_idx = start_idx + increment - 1
            ore_map = reshape(vals[start_idx:end_idx], p.n)
            # ore_map = repeat(poro_2D, outer=(1, 1, p.grid_dims[3]))
            push!(ore_quals, ore_map)
        end
        return ore_quals
    end
end

Base.rand(rng::Random.AbstractRNG, p::GSLIBDistribution, n::Int64=1, dir::String="sgsim_output/") = Base.rand(p, n, dir)

function kriging(p::GSLIBDistribution, obs::RockObservations, dir::String="kb2d_output/"; silent::Bool=true)
    if silent
        stdout_orig = stdout
        (rd, wr) = redirect_stdout()
    end
    fn = write_kb2d_params_to_file(p, obs; dir=dir) # NOTE: If we are going to want to sample many instances then we can include an "N" parameter here instead of the 1, but would need to update the code below as well

    # Run sgsim
    run(`kb2d $fn`)

    # Load the results and return
    vals = CSV.File("$(dir)kb2d.out", skipto=5, header=false, delim=" ", silencewarnings=true, select=[1,6]) |> CSV.Tables.matrix
    # reshape(vals, p.n..., N) # For multiple samples
    idxs = isa.(vals[:, 2], Missing)
    vals[idxs, 2] .= 1.0
    μ = zeros(Float64, p.grid_dims[1], p.grid_dims[2])
    σ² = zeros(Float64, p.grid_dims[1], p.grid_dims[2])
    μ[:, :] = reshape(vals[:, 1], p.grid_dims[1], p.grid_dims[2])
    σ²[:, :] = reshape(vals[:, 2], p.grid_dims[1], p.grid_dims[2])
    if silent
        redirect_stdout(stdout_orig)
        close(rd)
        close(wr)
    end
    return (μ, σ²)
end
