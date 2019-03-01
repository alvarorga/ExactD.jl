using ExactD
using Test

@testset "states with N particles and L sites" begin
    states = ExactD._get_LN_states(4, 2)

    @test states[1] == 3
    @test states[3] == 6
    @test states[5] == 10
    @test states[6] == 12
end

@testset "dense many-body operator" begin
    L = 4
    N = 2
    J::Array{Float64, 2} = reshape(collect(1:16), (4, 4))
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
end
