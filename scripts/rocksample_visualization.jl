using Compose

function POMDPTools.render(pomdp::RockSamplePOMDP, step;
            viz_rock_state=true,
            viz_belief=true,
            pre_act_text="",
            network=nothing,
        )
    nx, ny = pomdp.map_size[1] + 1, pomdp.map_size[2] + 1
    cells = []
    for x in 1:nx-1, y in 1:ny-1
        ctx = cell_ctx((x, y), (nx, ny))
        cell = compose(ctx, Compose.rectangle(), fill("white"))
        push!(cells, cell)
    end
    grid = compose(context(), linewidth(0.1mm), Compose.stroke("black"), cells...)
    outline = compose(context(), linewidth(1mm), Compose.rectangle(), Compose.stroke("black"))

    rocks = []
    for (i, (rx, ry)) in enumerate(pomdp.rocks_positions)
        ctx = cell_ctx((rx, ry), (nx, ny))
        clr = "black"
        if viz_rock_state && get(step, :s, nothing) !== nothing
            clr = step[:s].rocks[i] ? "green" : "red"
        end
        rock = compose(ctx, ngon(0.5, 0.5, 0.3, 6), linewidth(0.75mm), Compose.stroke(clr), fill("gray"))
        push!(rocks, rock)
    end
    rocks = compose(context(), rocks...)
    exit_area = render_exit((nx, ny))

    agent = nothing
    action = nothing
    if get(step, :s, nothing) !== nothing
        agent_ctx = cell_ctx(step[:s].pos, (nx, ny))
        agent = render_agent(agent_ctx)
        if get(step, :a, nothing) !== nothing
            action = render_action(pomdp, step)
        end
    end
    action_text = render_action_text(pomdp, step, pre_act_text; network)

    belief = nothing
    if viz_belief && (get(step, :b, nothing) !== nothing)
        belief = render_belief(pomdp, step)
    end
    sz = min(w, h)
    return compose(context((w - sz) / 2, (h - sz) / 2, sz, sz), action, agent, belief, rocks, action_text, grid, exit_area, outline)
end

function cell_ctx(xy, size)
    nx, ny = size
    x, y = xy
    return context((x - 1) / nx, (ny - y - 1) / ny, 1 / nx, 1 / ny)
end

function render_belief(pomdp::RockSamplePOMDP, step)
    rock_beliefs = get_rock_beliefs(pomdp, get(step, :b, nothing))
    nx, ny = pomdp.map_size[1] + 1, pomdp.map_size[2] + 1
    belief_outlines = []
    belief_fills = []
    for (i, (rx, ry)) in enumerate(pomdp.rocks_positions)
        ctx = cell_ctx((rx, ry), (nx, ny))
        clr = "black"
        belief_outline = compose(ctx, Compose.rectangle(0.1, 0.87, 0.8, 0.07), Compose.stroke("gray31"), fill("gray31"))
        belief_fill = compose(ctx, Compose.rectangle(0.1, 0.87, rock_beliefs[i] * 0.8, 0.07), Compose.stroke("lawngreen"), fill("lawngreen"))
        push!(belief_outlines, belief_outline)
        push!(belief_fills, belief_fill)
    end
    return compose(context(), belief_fills..., belief_outlines...)
end

function get_rock_beliefs(pomdp::RockSamplePOMDP{K}, b) where K
    rock_beliefs = zeros(Float64, K)
    for (sᵢ, bᵢ) in weighted_iterator(b)
        rock_beliefs[sᵢ.rocks.==1] .+= bᵢ
    end
    return rock_beliefs
end

function render_exit(size)
    nx, ny = size
    x = nx
    y = ny
    ctx = context((x - 1) / nx, (ny - y) / ny, 1 / nx, 1 - 1/nx)
    txt = compose(ctx, Compose.text(9, 0.535, "EXIT", hcenter, vcenter, Rotation(π/2)),
        Compose.stroke("white"),
        fill("white"),
        fontsize(12pt),
        Compose.font("Palatino Linotype"))
    return compose(ctx, txt, Compose.rectangle(), fill("darkred"))
end

function render_agent(ctx)
    center = compose(context(), Compose.circle(0.5, 0.5, 0.15), fill("orange"), Compose.stroke("black"))
    return compose(ctx, center)
end

function render_action_text(pomdp::RockSamplePOMDP, step, pre_act_text; network=nothing)
    actions = ["• Sample •", "North ↑", "East →", "South ↓", "West ←"]
    action_text = "Terminal"
    if get(step, :a, nothing) !== nothing
        if step.a <= RockSample.N_BASIC_ACTIONS
            action_text = actions[step.a]
        else
            action_text = "~ Sensing Rock $(step.a - RockSample.N_BASIC_ACTIONS) ~"
        end
        if !isnothing(network)
            v, p = network_lookup(network, step.b)
            v = round(v, digits=2)
            pa = round(p[step.a], digits=2)
            action_text = "$action_text ($v, $pa)"
        end
    end
    action_text = pre_act_text * action_text

    _, ny = pomdp.map_size
    ny += 1
    ctx = context(0, (ny - 1) / ny, 1, 1 / ny)
    txt = compose(ctx, Compose.text(0.5, -0.8, action_text, hcenter),
        Compose.stroke("black"),
        fill("black"),
        fontsize(10pt),
        Compose.font("Palatino Linotype"))
    return compose(ctx, txt, Compose.rectangle(), Compose.stroke("black"), fill("white"))
end

function render_action(pomdp::RockSamplePOMDP, step)
    if step.a == RockSample.BASIC_ACTIONS_DICT[:sample]
        ctx = cell_ctx(step.s.pos, pomdp.map_size .+ (1, 1))
        if in(step.s.pos, pomdp.rocks_positions)
            rock_ind = findfirst(isequal(step.s.pos), pomdp.rocks_positions)
            clr = step.s.rocks[rock_ind] ? "lightgreen" : "yellow"
        else
            clr = "black"
        end
        return compose(ctx, ngon(0.5, 0.5, 0.2, 6), Compose.stroke("gray"), fill(clr))
    elseif step.a > RockSample.N_BASIC_ACTIONS
        rock_ind = step.a - RockSample.N_BASIC_ACTIONS
        rock_pos = pomdp.rocks_positions[rock_ind]
        nx, ny = pomdp.map_size[1] + 1, pomdp.map_size[2] + 1
        rock_pos = ((rock_pos[1] - 0.5) / nx, (ny - rock_pos[2] - 0.5) / ny)
        rob_pos = ((step.s.pos[1] - 0.5) / nx, (ny - step.s.pos[2] - 0.5) / ny)
        sz = min(w, h)
        return compose(context((w - sz) / 2, (h - sz) / 2, sz, sz), line([rob_pos, rock_pos]), Compose.stroke("orange"), linewidth(0.01w))
    end
    return nothing
end


function rocksample_policy_plot(p; colors=vcat(:black, fill(:orange,4), fill(:gray, 15)))
    xticklabels = (1:20, string.(vcat(:x, :N, :E, :S, :W, 1:15)))
    return Plots.bar(p, c=colors, xticks=xticklabels, label=false, size=(600,300))
end