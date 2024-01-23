using RemoteJobs
machine_specs = [("localhost", 128, 8)]
sync_code(machine_specs, abspath(joinpath(@__DIR__, "..")))
launch_remote_workers(machine_specs)
