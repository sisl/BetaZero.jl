using RemoteJobs
machine_specs = [("mossr@tula.stanford.edu", 25), ("mossr@jodhpur.stanford.edu", 25)]
sync_code(machine_specs, abspath(joinpath(@__DIR__, "..")))
launch_remote_workers(machine_specs)
