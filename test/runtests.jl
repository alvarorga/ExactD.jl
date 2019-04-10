using ExactD
using Test

@testset "states with N particles and L sites" begin
    L = 4
    N = 2
    states = ExactD._get_LN_states(L, N)

    @test states[1] == 3
    @test states[3] == 6
    @test states[5] == 10
    @test states[6] == 12
end

@testset "spin 1 states with L spins and Sz magnetization" begin
    L = 4
    Sz = 2
    states = ExactD._get_spin1_LSz_states(L, Sz)

    @test length(states) == 10
    @test states[1] == 5
    @test states[3] == 20
    @test states[5] == 68
    @test states[7] == 86
    @test states[10] == 149

    L = 4
    Sz = 1
    states = ExactD._get_spin1_LSz_states(L, Sz)

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
    Jz = collect(1.:L)/2
    Jz2 = collect(1.:L)/4
    C = 1.33
    Op = build_spin1_many_body_op(L, Sz, J, Jz, Jz2, C)

    # Diagonal terms: (S^z + S^z)^2.
    @test Op[1, 1] ≈ 1.5 + 0.75 + C
    @test Op[4, 4] ≈ 2.5 + 1.25 + C
    @test Op[7, 7] ≈ 4. + 2.5 + C
    @test Op[10, 10] ≈ 1. + 2.5 + C

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
    Jz = collect(1.:L)/4
    Jz2 = collect(1.:L)/2
    Op = build_spin1_many_body_op(L, Sz, J, Jz, Jz2)
    # Off-diagonal terms.
    @test Op[4, 2] ≈ 12.
    @test Op[4, 5] ≈ 4.
end

@testset "type of many-body operators" begin
    L = 4
    N = 2
    C = complex(1.)
    J0::Array{ComplexF64, 2} = reshape(collect(1:L^2), (L, L))
    Op = build_many_body_op(L, N, J0, C)
    @test eltype(Op) == ComplexF64

    L = 4
    N = 2
    C = 1.
    J1::Array{Float64, 2} = reshape(collect(1:L^2), (L, L))
    Op = build_many_body_op(L, N, J1, C)
    @test eltype(Op) == Float64

    L = 4
    N = 2
    C = 0.1
    J2::Array{Float64, 2} = reshape(collect(1:L^2), (L, L))
    Op = build_many_body_op(L, N, J2, C)
    @test eltype(Op) == Float64

    L = 4
    N = 2
    J3::Array{Float64, 2} = reshape(collect(1:L^2), (L, L))
    Op = build_sparse_many_body_op(L, N, J3)
    @test eltype(Op) == Float64

    L = 3
    Sz = 0
    J4::Array{ComplexF64, 2} = reshape(collect(1:L^2), (L, L))
    Jz2 = complex.(collect(1.:L)/4)
    Jz = complex.(collect(1.:L)/2)
    Op = build_spin1_many_body_op(L, Sz, J4, Jz, Jz2)
    @test eltype(Op) == ComplexF64

    L = 3
    Sz = 0
    J5::Array{Float64, 2} = reshape(collect(1.:L^2), (L, L))
    Jz2 = collect(1.:L)/4
    Jz = collect(1.:L)/2
    Op = build_spin1_many_body_op(L, Sz, J5, Jz, Jz2)
    @test eltype(Op) == Float64
end
