# using FractalMC
# import FractalMC: Simulate!, ExecuteAction!, Reward, Distance, Actions
using Plots
using Random, Distributions
using Pkg
GR.inline("png")

""" Helper function to create covariance matrices """
function DiagonalMatrix(coef)
    [rand()*coef 0.; 0. rand()*coef]
end

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

struct GridModel <: AbstractModel
    grid::Array{Float64,2}
end

Actions(m::GridModel) = 1:4

function ExecuteAction!(m::GridModel, state::Array{Int,1}, a::Int)
    if a == 1 && state[1] < size(m.grid,1)
        state[1] += 1
    elseif a == 2 && state[2] < size(m.grid,2)
        state[2] += 1
    elseif a == 3 && state[1] > 1
        state[1] -= 1
    elseif a == 4 && state[2] > 1
        state[2] -= 1
    end
end

ExecuteAction!(m::GridModel, w::AbstractWalker, a::Int) = ExecuteAction!(m, w.state, a)

function Simulate!(e::StepEnvironment{GridModel}, w::AbstractWalker, dt::Int)
    w.reward = 0.
    statesHistory = []
    for i in 1:dt
        action = rand(Actions(e.model))
        ExecuteAction!(e.model,w,action)
        w.reward += Reward(e.model,w)
        push!(statesHistory, copy(w.state))
    end
    # w.reward = Reward(e.model, w)
    statesHistory
end

Reward(m::GridModel, s::Array{Int}) = m.grid[s[1],s[2]]
Reward(m::GridModel, w::AbstractWalker) = m.grid[w.state[1], w.state[2]]
Distance(m::GridModel, w1::AbstractWalker, w2::AbstractWalker) = sum(abs.(w1.state-w2.state))

Random.seed!(0)
grid = GenerateGrid(100,100,100,150)
model = GridModel(grid)
env = StepEnvironment(BasicWalker, 200, model, [10,10], 100, 10)

# function ScanLoopHook(e::StepEnvironment{<:AbstractWalker,GridModel})
#     println("Countmap $(countmap(map(x->x.initialDecision,env.walkers)))")
# end
Scan!(env)
maximum(map(x->x.reward, env.walkers))
countmap(map(x->x.state, env.walkers))

unique(map(x->x.initialDecision,env.walkers))


function TempDecide(env)
    countPerAction = countmap(map(x->x.initialDecision,env.walkers))
    rewardPerAction = zeros(4)
    for walker in env.walkers
        dec = walker.initialDecision
        rewardPerAction[dec] += walker.reward / countPerAction[dec]
    end

    argmax(rewardPerAction)
end

TempDecide(env)
states = []
for i in 1:100
    push!(states, copy(env.initialState))
    Scan!(env)
    # println("Countmap $(countmap(map(x->x.initialDecision,env.walkers)))")
    decision = TempDecide(env)
    ExecuteAction!(env.model, env.initialState, decision)
    println("Decision: $decision - Reward: $(Reward(env.model, env.initialState))")

    ResetWalkers!(env)
    # println("Init decision $(countmap(map(x->x.initialDecision,env.walkers)))")
end


img = @gif for i in 1:size(states,1)
    heatmap(env.model.grid, legend=false)
    scatter!((states[i][2], states[i][1]))
    annotate!( [(1,3,Plots.text("Step $i", 7, :left, :white))] )
end

map(w->Simulate!(env, w), env.walkers )
map(x->x.state, env.walkers)

heatmap(grid, legend=false)

function PlotWalkersTrace(env::StepEnvironment{GridModel}, log)
    n_walkers = size(env.walkers,1)
    n_steps = size(log[1],1)
    img = @gif for i in 1:n_steps
        heatmap(env.model.grid, legend=false)
        scatter!(map(x->(x[i][2], x[i][1]), log))
        annotate!( [(1,3,Plots.text("Step $i", 7, :left, :white))]
    end every 5
    img
end

PlotWalkersTrace(env, Scan!(env))
