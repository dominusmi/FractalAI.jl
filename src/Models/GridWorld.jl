module GridWorld

import ..FractalAI: AbstractModel, AbstractWalker, Actions, Reward, ExecuteAction!,
                    Simulate!, Distance, StepEnvironment

export GridModel

struct GridModel <: AbstractModel
    grid::Array{Float64,2}
end

""" Actions must return an object <: AbstractArray """
Actions(m::GridModel) = 1:4

function ExecuteAction!(m::GridModel, w::AbstractWalker, a::Int)
    if a == 1 && w.state[1] < size(m.grid,1)
        w.state[1] += 1
    elseif a == 2 && w.state[2] < size(m.grid,2)
        w.state[2] += 1
    elseif a == 3 && w.state[1] > 1
        w.state[1] -= 1
    elseif a == 4 && w.state[2] > 1
        w.state[2] -= 1
    end
end

""" Simulates dt time steps for walker w. Returns log of states """
function Simulate!(e::StepEnvironment{GridModel}, w::AbstractWalker, dt::Int)
    w.reward = 0.
    statesHistory = []
    for i in 1:dt
        # Pick action at random and execute
        action = rand(Actions(e.model))
        ExecuteAction!(e.model,w,action)

        # Doing cumulative reward per interval
        w.reward += Reward(e.model,w)

        # Log state
        push!(statesHistory, copy(w.state))
    end
    statesHistory
end

# Reward is simply value of grid at walker's position
Reward(m::GridModel, w::AbstractWalker) = m.grid[w.state[1], w.state[2]]
# Manhattan (L1) distance
Distance(m::GridModel, w1::AbstractWalker, w2::AbstractWalker) = sum(abs.(w1.state-w2.state))

""" Helper function to create covariance matrices """
function DiagonalMatrix(coef)
    [rand()*coef 0.; 0. rand()*coef]
end

""" Generates a grid """
function GenerateGrid(x_length, y_length, n_sources=x_length, n_samples=x_length)
    x_int = 1:x_length
    y_int = 1:y_length

    grid = zeros(x_length, y_length)

    for i in 1:n_sources
        μ = rand(2) .* [x_length, y_length]
        Σ = DiagonalMatrix(x_length*0.5)
        dist = MvNormal(μ, Σ)

        samples = rand(dist, n_samples)
        samples = round.(Integer, samples)

        for sample in eachcol(samples)
            if sample[1] < 1 || sample[1] > x_length
                continue
            elseif sample[2] < 1 || sample[2] > y_length
                continue
            end
            grid[sample[1], sample[2]] += 1
        end
    end
    grid ./= sum(grid)
    grid
end


end # module
