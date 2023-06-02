using RemoteJobs
machine_specs = [("user@host1", 25), ("user@host2", 25)]
sync_code(machine_specs, abspath(joinpath(@__DIR__, "..")))
launch_remote_workers(machine_specs)
