include(joinpath(@__DIR__, "..", "remote.jl"))
machine_specs = [("mossr@oceanside.stanford.edu", 40), ("mossr@jodhpur.stanford.edu", 30), ("mossr@tver.stanford.edu", 30)]
sync_code(machine_specs)
launch_remote_workers(machine_specs)
