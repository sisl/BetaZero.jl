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
            if length(ms) == 3 # use threads
                use_threads = true
                host, n, n_threads = ms
            else
                use_threads = false
                host, n = ms
            end
            @info "Adding $n processes on $host..."
            if use_threads
                # cpu_threads = parse(Int,strip(read(`ssh $host 'julia --print "Sys.CPU_THREADS"'`, String)))
                # n_threads = cpu_threads ÷ n
                @info "With $n_threads threads per process on $host..."
                threads_args = "--threads=$n_threads"
                kwargs = (exeflags=[threads_args],)
            else
                kwargs = ()
            end
            if host == "localhost"
                addprocs(n; kwargs...)
            else
                addprocs([(host, n)]; tunnel=true, kwargs...)
            end
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
        if host != "localhost"
            @info "Syncing code to $host:$dir..."
            run(Cmd(["sh", "-c", "scp -r $dir/* $host:$dir/"]))
        end
    end
end

end # module RemoteJobs
