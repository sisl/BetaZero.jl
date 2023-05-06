using Suppressor
using ProgressMeter

function write_tikz_datatable(XY, Z, filename="drills.dat";
                             dir=joinpath(@__DIR__, "..", "tex", "data"),
							 xmax=maximum(first.(XY)),
							 ymax=maximum(last.(XY)),
							 zmax=maximum(Z),
							 xmin=minimum(first.(XY)),
							 ymin=minimum(last.(XY)),
							 zmin=minimum(Z),
							 scale=1)
	tab = " "^3
	table = """
	X$(tab)Y$(tab)Z
	$xmin$tab$ymin$tab$(scale*zmin)$tab%min
	$xmax$tab$ymax$tab$(scale*zmax)$tab%max
	"""
	for ((x,y),z) in zip(XY, Z)
		table = string(table, "\n", x, tab, y, tab, scale*z)
	end
    open(joinpath(dir, filename), "w+") do f
        write(f, table)
    end
end


function write_tikz_decisiontable(p_decisions::Vector, filename="decisions.dat"; dir=joinpath(@__DIR__, "..", "tex", "data"))
	open(joinpath(dir, filename), "w+") do f
        write(f, "NO   YES\n$(p_decisions[1])   $(p_decisions[2])")
    end
end


function write_tikz_actiontable(action, filename="action.dat"; dir=joinpath(@__DIR__, "..", "tex", "data"))
	if action isa Tuple
        x, y = action
        info = "drill"
    else
        x, y = -1, -1
        info = string(action)
    end
    open(joinpath(dir, filename), "w+") do f
        write(f, "X   Y   Info\n$x   $y   $info")
    end
end


function write_latex_data(pomdp::POMDP, policy::Union{BetaZeroPolicy,RawValueNetworkPolicy}, belief, action; use_mean=true, dir=joinpath(@__DIR__, "..", "tex", "data"))
    A = POMDPs.actions(pomdp)
    Ab = POMDPs.actions(pomdp, belief)
    p = BetaZero.policy_lookup(policy, belief)
    XY = A[3:end] # MinExPOMDP: first two actions are [:abandon, :mine], rest are x/y drill locations

    # Zero-out already selected actions
    tried_actions = setdiff(A, Ab)
    if !isempty(tried_actions)
        for (i,a) in enumerate(A)
            for a_tried in tried_actions
                if a == a_tried
                    p[i] = 0 # zero-out
                    break
                end
            end
        end
    end
    p_decisions = p[1:2]
    p_xy = p[3:end]

    !isdir(dir) && mkdir(dir)
    write_tikz_datatable(XY, p_xy, "drills.dat"; xmax=pomdp.grid_dims[1], ymax=pomdp.grid_dims[2], scale=50, dir)
    b̃ = BetaZero.input_representation(belief)
    if use_mean
        bmap = b̃[:,:,1] # mean
    else
        bmap = b̃[:,:,2] # std
    end
    write_tikz_datatable([(i,j) for i in 1:pomdp.grid_dims[1], j in 1:pomdp.grid_dims[2]], bmap, "map.dat"; dir)
    write_tikz_decisiontable(p_decisions; dir)
    write_tikz_actiontable(action; dir)
    return nothing
end


function tikz_policy_plots(pomdp::POMDP, policy::Union{BetaZeroPolicy,RawValueNetworkPolicy}, beliefs::Vector, actions::Vector; clean=true, kwargs...)
    @info "Creating $(length(beliefs)) TiKZ policy maps."
    dir = joinpath(@__DIR__, "..", "tex")

    if clean
        cd(dir) do
            for f in readdir()
                if !isnothing(match(r"policy_map\d\.pdf", f))
                    rm(f) # clean old files
                end
            end
        end
    end

    @suppress_out @showprogress for (i,belief) in enumerate(beliefs)
        write_latex_data(pomdp, policy, belief, actions[i]; kwargs...)
        cd(dir) do
            run(`pdflatex policy_map.tex`)
            mv("policy_map.pdf", "policy_map$i.pdf", force=true)
        end
    end
end
