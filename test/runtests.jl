using ExactD
using Test

include("../src/states.jl")

@testset "states with N particles and L sites" begin
    states = _get_states(4, 2)

    @test states[1] == 3
    @test states[3] == 6
    @test states[5] == 10
    @test states[6] == 12
end # testset
