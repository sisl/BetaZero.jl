# ENV["CUDA_VISIBLE_DEVICES"] = "1"
using RemoteJobs
# machine_specs = [("mossr@oceanside.stanford.edu", 25), ("mossr@jodhpur.stanford.edu", 25), ("mossr@tula.stanford.edu", 25), ("mossr@tver.stanford.edu", 25)]
# machine_specs = [("mossr@oceanside.stanford.edu", 25), ("mossr@jodhpur.stanford.edu", 25), ("mossr@tver.stanford.edu", 25)]
# machine_specs = [("mossr@oceanside.stanford.edu", 25), ("mossr@jodhpur.stanford.edu", 25)]
# machine_specs = [("mossr@oceanside.stanford.edu", 40), ("mossr@jodhpur.stanford.edu", 30), ("mossr@tver.stanford.edu", 30)]
# machine_specs = [("mossr@oceanside.stanford.edu", 80), ("mossr@jodhpur.stanford.edu", 40)]
machine_specs = [("mossr@tula.stanford.edu", 25), ("mossr@jodhpur.stanford.edu", 25)]
sync_code(machine_specs, abspath(joinpath(@__DIR__, "..")))
launch_remote_workers(machine_specs)
