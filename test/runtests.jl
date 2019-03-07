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
    J::Array{Float64, 2} = reshape(collect(1:16), (L, L))
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
    J::Array{Float64, 2} = reshape(collect(1:16), (L, L))
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
