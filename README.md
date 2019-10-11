ExactD.jl
====

Exact diagonalization of Hamiltonians in Julia. ExactD.jl can build many-body operators for hard-core boson and spin-1 systems of one- and two-body operators. It can also compute expectation values of many-body operators given a many-body vector state.

Many-body operators
---

Usually one starts by defining a basis of Fock states upon which the many-body operators act: 
* For sistems with two degrees of freedom, e.g. spin-1/2, hard-core bosons or fermions: `make_LN_basis(L, N)` makes a basis of all states with `L` sites and `N` particles. 
* For spin-1 systems: `make_spin1_LSz_basis(L, Sz)` makes basis of all states with `L` spins and total magnetization `Sz`. The single states that a single spin can have are `|1,+1>`, `|1,0>`, and `|1,-1>`.

 ### Hard-core bosons.

 One can build many-body operators of Hamiltonians like
 
 ` H = C + \sum_{ij} J_{ij} b^\dagger_i b_j + V_{ij} n_i n_j `
 
 The method used to build the many-body Hamiltonian is
 ```julia
 build_many_body_op(L::Int,
                    basis::Vector{Int},
                    J::AbstractMatrix{T},
                    V::AbstractMatrix{T},
                    C::T=zero(T)) where T<:Number
 ```
 Additionally, for very large basis one can build sparse many-body operators using `build_sparse_many_body_op`, with the same argument syntax as `build_many_body_op`.

 ### Spin-1.

 One can build many-body operators of spin-1 Hamiltonians like

 ` H = C + \sum_i Jz_i S^z_i + \sum{i\neq j} J_{ij} S^+_i S^-_j + \sum_{i\leq j=1} W_{ij} S^z_i S^z_j `

The method used to build this many-body Hamiltonian is

 ```julia
 build_spin1_many_body_op(L::Int,
                          basis::Vector{Int},
                          J::AbstractMatrix{T},
                          W::AbstractMatrix{T},
                          Jz::Vector{T},
                          C::T=zero(T)) where T<:Number
 ```

Expectation values
---

This is only implemented for systems with two degrees of freedom per site. Usually one has already defined a `basis` of Fock states and has computed the wavefunction vector `state`.

* To compute expectation values of strings of number operators like `<state|n_p[1]*n_p[2]*...*n_p[n]|state>` one should use the function

```julia
expected_n(basis::Vector{Int},
           state::Vector{<:Number},
           p::Vector{Int})
```
For only one number operator there is defined `expected_n(basis::Vector{Int}, state::Vector{<:Number}, p::Int)`.

* To compute expectation values of strings of normal ordered creation and annihilation operators like `<state|b^+_p[1]*...*b^+_p[n]*b_q[1]*...*b_q[m]|state>` one should use the function

```julia
expected_pq(basis::Vector{Int},
            state::Vector{<:Number},
            p::Vector{Int},
            q::Vector{Int})
```
For one-body correlations there is defined `expected_pq(basis::Vector{Int}, state::Vector{<:Number}, p::Int, q::Int)`.


Entanglement entropy
---

Also for systems with two degrees of freedom per site there is a function that computes the entanglement entropy of a given `state` wavefunction at a site `i` in the lattice.

```julia
get_entanglement_entropy(L::Int,
                         i::Int,
                         basis::Vector{Int},
                         state::Vector{<:Number})
```
