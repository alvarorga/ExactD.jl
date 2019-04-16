ExactD.jl
====

Exact diagonalization of Hamiltonians in Julia

Usage
---

ExactD.jl can build many-body operators for hard-core boson  and spin-1 systems given a hopping matrix and diagonal terms.

 ### Hard-core bosons.

 This part of the module allows you to diagonalize Hamiltonians of the form

 `H = C + \sum^L_{ij=1} J_{ij} b^\dagger_i b_j`

 with `L` the number of sites in the system, `J` the hopping matrix, `C` a constant term, and you have to choose the number of particles `N` in the system. The functions to build dense and sparse many-body operators are

 ```julia
 build_many_body_op(L::Int, N::Int, J::Array{T, 2}, C::T=zero(T))
 build_sparse_many_body_op(L::Int, N::Int, J::Array{T, 2}, C::T=zero(T))
 ```

 ### Spin-1.

 This module also allows you to diagonalize spin-1 systems of the form

 ` H = C + \sum^L_{i=1} Jz_i S^z_i + \sum^L_{i\neq j=1} J_{ij} S^+_i S^-_j + \sum^L_{i\le j=1} W_{ij} S^z_i S^z_j`

 with the allowed states in each site of the system: `|1,+1>`, `|1,0>`, and `|1,-1>`. The function to build dense spin-1 many-body operators with magnetization `Sz` is

 ```julia
 build_spin1_many_body_op(L::Int, Sz::Int, J::Array{T, 2}, W::Array{T, 2}, Jz::Vector{T}, C::T=zero(T))
 ```
