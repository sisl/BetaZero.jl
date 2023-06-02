"""
Light-weight package to launch jobs across computers and sync code.
"""
module RemoteJobs

using Reexport
@reexport using Distributed
export launch_remote_workers, sync_code


"""
Launch remote workers using `machine_specs` list of `[("user@host", num_processes), ...]`
"""
function launch_remote_workers(machine_specs::Vector)
    total_procs_requested = sum(ms[2] for ms in machine_specs)
    if nprocs() < total_procs_requested + 1 # Include main process (+1)
        for ms in machine_specs
            host, n = ms
            @info "Adding $n processes on $host..."
            addprocs([(host, n)], tunnel=true)
        end
        @info "Finished launching processes."
    else
        @info "Skipped launching processes, already launched."
    end
end


"""
Sync code from main server to other host servers.
"""
function sync_code(machine_specs::Vector, dir::String)
    for ms in machine_specs
        host, _ = ms
        @info "Syncing code to $host:$dir..."
        run(Cmd(["sh", "-c", "scp -r $dir/* $host:$dir/"]))
    end
end

end # module RemoteJobs
