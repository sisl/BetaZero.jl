module RobotLocalization

using RoombaPOMDPs
using POMDPs
using POMDPTools
using ParticleFilters
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using Statistics

export
    RobotPOMDP,
    RobotMDP,
    RobotParticleFilter,
    RobotState,
    RobotAction

const RobotPOMDP = RoombaPOMDP
const RobotMDP = RoombaMDP
const RobotParticleFilter = RoombaParticleFilter
const RobotState = RoombaState
const RobotAction = RoombaAct


function RobotPOMDP(; config=1, goal_reward=1000, stairs_penalty=-goal_reward)
    aspace = vec(collect(RoombaAct(v, ω) for v in [0, 2], ω in [-2, 0, 2]))
    sensor = Lidar()
    mdp = RoombaMDP(; config, aspace, goal_reward, stairs_penalty)
    return RoombaPOMDP(sensor, mdp)
end

function POMDPs.reward(pomdp::RobotPOMDP, b::ParticleCollection, a::RobotAction, bp::ParticleCollection)
    return mean(reward(pomdp, s, a, sp) for (s,sp) in zip(particles(b), particles(bp)))
end

function POMDPs.updater(pomdp::RobotPOMDP; n_particles=1000, vel_noise=2.0, ω_noise=0.5)
    return RoombaParticleFilter(pomdp, n_particles, vel_noise, ω_noise)
end

function POMDPs.update(up::RoombaParticleFilter, b::ParticleCollection, a, o)
    pm = up._particle_memory
    wm = up._weight_memory
    empty!(pm)
    empty!(wm)
    all_terminal = true
	handle_all_terminal = true
    for s in particles(b)
        if !POMDPs.isterminal(up.model, s) || !handle_all_terminal
            all_terminal = false
            # noise added here:
            a_pert = a + RoombaPOMDPs.SVec2(up.v_noise_coeff * (rand(up.rng) - 0.5), up.om_noise_coeff * (rand(up.rng) - 0.5))
            sp = @gen(:sp)(up.model, s, a_pert, up.rng)
            push!(pm, sp)
            push!(wm, obs_weight(up.model, s, a_pert, sp, o))
        end
    end
    if all_terminal && handle_all_terminal
	    # if all particles are terminal, issue an error
        # error("Particle filter update error: all states in the particle collection were terminal.")
		return deepcopy(b) # return previous belief if all particles are terminal
	else
	    return ParticleFilters.resample(up.resampler,
	                    WeightedParticleBelief(pm, wm, sum(wm), nothing),
	                    up.model,
	                    up.model,
	                    b, a, o,
	                    up.rng)
	end
end


Base.:(^)(s::RobotState, e::Real) = RobotState(s.x^e, s.y^e, s.theta^e, s.status^e)
Base.sqrt(s::RobotState) = s^(1/2)
Statistics.std(b::ParticleCollection{RobotState}) = sqrt(sum(map(s->s - mean(b), b.particles).^2)/length(b.particles))


function render(pomdp, steps, t; accuracy=(sp)->nothing, kwargs...)
	X = map(step->step.s.x, steps)
	Y = map(step->step.s.y, steps)

	Xb = map(s->s.x, particles(steps[t].b))
	Yb = map(s->s.y, particles(steps[t].b))

	μb = mean(steps[t].b)
	σb = std(steps[t].b)

	plot(legend=:bottomright)
	part_color = :black # :steelblue
	belief_color = :gold
	agent_color = :firebrick
	lidar_color = :tomato
	traj_color = :black # :black
	scatter!([], [], c=part_color, msc=part_color, α=0.75, lab="belief particles")
	scatter!(Xb, Yb, c=part_color, msc=part_color, ms=1.5, α=0.2, lab=false)
	scatter!([μb.x], [μb.y], c=belief_color, m=:star4, ms=6, α=1.0, lab="belief mean")
	plot!(X, Y, marker=true, ms=2, mc=:white, c=traj_color, lab="trajectory", α=1)

	s = steps[t].s
	o = steps[t].o
	θ = s.theta
	num_points = 3round(Int,o)
	lidar_x = clamp(s.x + o*cos(θ), -Inf, Inf) # 25.5, 15.5) # TODO
	lidar_y = clamp(s.y + o*sin(θ), -Inf, Inf) # 20.5, 5.5) # TODO
	scatter!(range(s.x, lidar_x, num_points), range(s.y, lidar_y, num_points), ls=:dot, ms=1, c=lidar_color, msc=lidar_color, label=false)
	scatter!([lidar_x], [lidar_y], c=lidar_color, ms=2, label="lidar observation")
	# scatter!([X[t]], [Y[t]], c=agent_color, marker=:circle, label="agent position")
	annotate!(X[t], Y[t], Plots.text("➤", 16, :white, rotation=rad2deg(θ))) # outline
	annotate!(X[t], Y[t], Plots.text("➤", 11, agent_color, rotation=rad2deg(θ)))
	scatter!([], [], c=agent_color, msc=agent_color, ms=2, marker=:rtriangle, label="agent position")

	γ = discount(pomdp)
	R = sum(γ^(t′-1)*steps[t′].r for t′ in 1:length(steps))
	rd = x->round(x; digits=2)
	s = steps[t].s
	textx = -13
	texty = -7
	annotate!(textx, texty, ("\$t=$t\$", 12, :black, :left))
	annotate!(textx, texty-2, ("\$x=$(rd(s.x))\$", 12, :black, :left))
	annotate!(textx, texty-4, ("\$y=$(rd(s.y))\$", 12, :black, :left))
	annotate!(textx, texty-6, ("\$\\theta=$(rd(s.theta))\$", 12, :black, :left))
	annotate!(textx, texty-10, ("\$\\operatorname{reward}=$(rd(steps[t].r))\$", 12, :black, :left))
	correct = accuracy(steps[end].sp)
	if !isnothing(correct)
		correct_text = correct ? "✓" : "×"
		annotate!(textx, texty-12, ("\$\\operatorname{return}=$(rd(R))\$ [$correct_text]", 12, :black, :left))
	end

	annotate!(textx+10, texty-2, ("\$(b_x=$(rd(μb.x)) \\pm $(rd(σb.x)))\$", 12, :black, :left))
	annotate!(textx+10, texty-4, ("\$(b_y=$(rd(μb.y)) \\pm $(rd(σb.y)))\$", 12, :black, :left))
	annotate!(textx+10, texty-6, ("\$(b_\\theta=$(rd(μb.theta)) \\pm $(rd(σb.theta)))\$", 12, :black, :left))

	plot!(framestyle=:none)
	draw_room(pomdp.mdp; kwargs...)
end


function draw_room(mdp; hold=true, from_gif=false)
	!hold && plot()
	for rectangle in mdp.room.rectangles
		segments = rectangle.segments
		for segment in segments
			x = [segment.p1[1], segment.p2[1]]
			y = [segment.p1[2], segment.p2[2]]
			if segment.goal
				c = :green3
				marker = :square
				ls = :dash
			elseif segment.stairs
				c = :red2
				marker = :square
				ls = :dash
			else
				c = :black
				marker = false
				ls = :solid
			end

			lw = (segment.goal || segment.stairs) ? (from_gif ? 1 : 2) : 2
			plot!(x, y, c=c, lw=lw, ls=ls, marker=marker, ms=0.95, mc=:black, label=false)
		end
	end
	return plot!(xlims=(-26, 16), ylims=(-21, 6))
end

end # module RobotLocalization
