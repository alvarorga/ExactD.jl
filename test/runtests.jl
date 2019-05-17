using ExactD
using Test

@testset "states with N particles and L sites" begin
    L = 4
    N = 2
    states = ExactD.get_LN_states(L, N)

    @test states[1] == 3
    @test states[3] == 6
    @test states[5] == 10
    @test states[6] == 12
end

@testset "spin 1 states with L spins and Sz magnetization" begin
    L = 4
    Sz = 2
    states = ExactD.get_spin1_LSz_states(L, Sz)

    @test length(states) == 10
    @test states[1] == 5
    @test states[3] == 20
    @test states[5] == 68
    @test states[7] == 86
    @test states[10] == 149

    L = 4
    Sz = 1
    states = ExactD.get_spin1_LSz_states(L, Sz)

    @test length(states) == 16
    @test states[1] == 1
    @test states[3] == 16
    @test states[5] == 25
    @test states[8] == 70
    @test states[10] == 82
    @test states[16] == 148
end

@testset "dense many-body operator" begin
    L = 4
    N = 2
    J::Array{Float64, 2} = reshape(collect(1:L^2), (L, L))
    Op = build_many_body_op(L, N, J)

    # Diagonal terms.
    @test Op[1, 1] ≈ 7.
    @test Op[4, 4] ≈ 17.
    @test Op[6, 6] ≈ 27.

    # Off-diagonal terms.
    @test Op[1, 2] ≈ 10.
    @test Op[2, 5] ≈ 0.
    @test Op[4, 2] ≈ 12.
    @test Op[6, 1] ≈ 0.
    @test Op[3, 5] ≈ 15.

    C = 1.3
    Op2 = build_many_body_op(L, N, J, C)

    # Diagonal terms.
    @test Op2[1, 1] ≈ 8.3
    @test Op2[4, 4] ≈ 18.3
    @test Op2[6, 6] ≈ 28.3
end

@testset "sparse many-body operator" begin
    L = 4
    N = 2
    J::Array{Float64, 2} = reshape(collect(1:L^2), (L, L))
    Op = build_sparse_many_body_op(L, N, J)

    # Diagonal terms.
    @test Op[1, 1] ≈ 7.
    @test Op[4, 4] ≈ 17.
    @test Op[6, 6] ≈ 27.

    # Off-diagonal terms.
    @test Op[1, 2] ≈ 10.
    @test Op[2, 5] ≈ 0.
    @test Op[4, 2] ≈ 12.
    @test Op[6, 1] ≈ 0.
    @test Op[3, 5] ≈ 15.

    C = 1.3
    Op2 = build_sparse_many_body_op(L, N, J, C)

    # Diagonal terms.
    @test Op2[1, 1] ≈ 8.3
    @test Op2[4, 4] ≈ 18.3
    @test Op2[6, 6] ≈ 28.3
end

@testset "dense many-body operator with L spins s=1" begin
    L = 4
    Sz = 2
    J = reshape(collect(1.:L^2), (L, L))
    W = reshape(collect(1.:L^2), (L, L))
    Jz = collect(1.:L)/2
    C = 1.33
    Op = build_spin1_many_body_op(L, Sz, J, W, Jz, C)

    # Diagonal terms: S^z + (S^z_i)^2 + S^z_i*S^z_j.
    @test Op[1, 1] ≈ 1.5 + 7 + 5 + C
    @test Op[4, 4] ≈ 2.5 + 17 + 13 + C
    @test Op[7, 7] ≈ 4. + 34 + 12 + C
    @test Op[10, 10] ≈ 1. + 34 - 18 + C

    # Off-diagonal terms.
    @test Op[1, 2] ≈ 20.
    @test Op[1, 4] ≈ 28.
    @test Op[1, 6] ≈ 0.
    @test Op[7, 6] ≈ 4.
    @test Op[6, 7] ≈ 10.
    @test Op[1, 9] ≈ 30.
    @test Op[9, 1] ≈ 24.
    @test Op[1, 10] ≈ 24.

    L = 3
    Sz = 0
    J = reshape(collect(1.:L^2), (L, L))
    W = reshape(collect(1.:L^2), (L, L))
    Jz = collect(1.:L)/4
    Op = build_spin1_many_body_op(L, Sz, J, W, Jz)
    # Off-diagonal terms.
    @test Op[4, 2] ≈ 12.
    @test Op[4, 5] ≈ 4.
end

@testset "type of many-body operators" begin
    L = 4
    N = 2
    C = complex(1.)
    J = complex.(reshape(collect(1.:L^2), (L, L)))
    Op = build_many_body_op(L, N, J, C)
    @test eltype(Op) == ComplexF64

    L = 4
    N = 2
    C = 1.
    J = reshape(collect(1.:L^2), (L, L))
    Op = build_many_body_op(L, N, J, C)
    @test eltype(Op) == Float64

    L = 4
    N = 2
    J = reshape(collect(1.:L^2), (L, L))
    Op = build_sparse_many_body_op(L, N, J)
    @test eltype(Op) == Float64

    L = 3
    Sz = 0
    J = complex.(reshape(collect(1.:L^2), (L, L)))
    W = complex.(reshape(collect(1.:L^2), (L, L)))
    Jz = complex.(collect(1.:L)/2)
    Op = build_spin1_many_body_op(L, Sz, J, W, Jz)
    @test eltype(Op) == ComplexF64

    L = 3
    Sz = 0
    J = reshape(collect(1.:L^2), (L, L))
    W = reshape(collect(1.:L^2), (L, L))
    Jz = collect(1.:L)/2
    Op = build_spin1_many_body_op(L, Sz, J, W, Jz)
    @test eltype(Op) == Float64
end
