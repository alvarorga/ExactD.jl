module ExactD

export
    build_many_body_op, build_sparse_many_body_op, build_spin1_many_body_op,
    get_num_spin1_states, get_LN_states, get_LN1N2_states,
    get_entanglement_entropy

using SparseArrays, LinearAlgebra

include("./states.jl")
include("./many_body_operators.jl")
include("./entropy.jl")
include("./auxiliary.jl")
include("./expected.jl")

end  # module ExactD
