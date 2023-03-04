using HDF5

save_states(mat::Array, filename="data.h5") = h5write(filename, "data", mat)
load_states(filename) = h5read(filename, "data")
