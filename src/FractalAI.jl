module FractalAI

using Statistics, StatsBase

export  AbstractWalker, AbstractModel, AbstractEnvironment, Simulate!, Actions,
        Reward, Distance, Scan!, Environment, Decide, BasicWalker, ExecuteAction!,
        StepEnvironment


abstract type AbstractWalker end

mutable struct BasicWalker{R<:Number,S,D} <: AbstractWalker
    reward::R
    state::S
    initialDecision::D
end
BasicWalker(state, action) = BasicWalker(0., state, action)

function Base.copy!(w1::BasicWalker, w2::BasicWalker)
    w1.state = copy(w2.state)
    w1.initialDecision = copy(w2.initialDecision)
end

"""
    A model needs to define the following functions:

    Actions(m): returns the actions available. Must be compatible with rand - rand(Actions(m))
    Simulate!(m, w, dt): Simulate dt events (updates walkers)
    Reward(m, w): returns the state reward
    Distance(m, w, w): returns the distance between walkers
    ExecuteAction!(m,w,action): executes the action in the environment

    where m::AbstractModel, and w::AbstractWalker
"""
abstract type AbstractModel end
abstract type DiscreteModel end
abstract type ContinuousModel end

abstract type AbstractState end

"""
    An environment acts as a wrapper that contains all the required simulation
    information, and requires:
        walkers
        model
        initialState
        dt
        τ
        log
"""
abstract type AbstractEnvironment{W,M,S} end

"""
    Basic environment where timesteps are assumed to be integer
"""
mutable struct StepEnvironment{M<:AbstractModel,W<:AbstractWalker,S<:Any} <: AbstractEnvironment{W,M,S}
    walkers::Array{W}
    model::M
    initialState::S
    dt::Int
    τ::Int
end

function StepEnvironment(walkersType::Type, n_walkers::Int, model::AbstractModel,
                    initialState::Any, τ::Int, dt::Int)

    actions = Actions(model)
    walkers = [walkersType(copy(initialState), rand(actions)) for i in 1:n_walkers]
    StepEnvironment(walkers, model, initialState, dt, τ)
end

""" Sets all walkers back to initial state and take initial decision """
function ResetWalkers!(e::AbstractEnvironment)
    actions = Actions(model)
    for i in 1:size(e.walkers,1)
        e.walkers[i].state = copy(e.initialState)
        e.walkers[i].initialDecision = rand(actions)
        e.walkers[i].reward = 0.
    end
end

Activation(x::AbstractFloat) = x < 0 ? exp(x) : 1+log(1+x)

function Relativise!(array::Array{<:AbstractFloat})
    μ = mean(array)
    σ = std(array, mean=μ)
    array .-= μ
    array ./= σ
    map!(Activation, array, array)
end

Actions(m::AbstractModel) = m.actions
Actions(e::AbstractEnvironment) = Actions(e.model)

ExecuteAction!(m::AbstractModel, w::AbstractWalker, a) = nothing

Simulate!(e::AbstractEnvironment, w::AbstractWalker, dt::Number) = nothing
Simulate!(e::AbstractEnvironment, w::AbstractWalker) = Simulate!(e, w, e.dt)

Reward(m::AbstractModel, w::AbstractWalker) = 0.
Reward(e::AbstractEnvironment, w::AbstractWalker) = Reward(e.model, w)

Distance(m::AbstractModel, w1::AbstractWalker, w2::AbstractWalker) = 0.
Distance(e::AbstractEnvironment, w1::AbstractWalker, w2::AbstractWalker) = Distance(e.model, w1, w2)
Distance(e::AbstractEnvironment, i::Int, j::Int) = Distance(e.model, e.walkers[i], e.walkers[j])

""" Calculates the virtual reward probability """
CloneProbability(VR_src::AbstractFloat, VR_dest::AbstractFloat) = (VR_dest-VR_src) / VR_src

ScanLoopHook(e::AbstractEnvironment) = nothing

""" Overwrite ith walker with jth """
function Clone!(e::AbstractEnvironment, i::Int, j::Int)
    copy!(e.walkers[i], e.walkers[j])
end

""" Returns a random number j != i in interval 1:k """
function DifferentIndexInInterval(i::Int, interval::AbstractArray)
    j = i
    while j == i
        j = rand(interval)
    end
    j
end

""" Computes virtual reward given current walker states """
function ComputeVirtualReward(e)
    n_walkers = size(e.walkers,1)
    R = zeros(n_walkers)
    D = zeros(n_walkers)

    for (i,walker) in enumerate(e.walkers)
        j = DifferentIndexInInterval(i, 1:n_walkers)
        R[i] = Reward(e, walker)
        D[i] = Distance(e, i, j)
    end

    Relativise!(R)
    Relativise!(D)

    R .* D
end

""" Run the scan process """
function Scan!(e::AbstractEnvironment)
    n_walkers = size(e.walkers,1)
    log = [[copy(e.initialState)] for i in 1:size(e.walkers,1)]

    # First step
    for (i,walker) in enumerate(e.walkers)
        statesHistory = Simulate!(e, walker)
        append!(log[i], statesHistory)
    end

    for t in e.dt:e.dt:e.τ
        ### Compute rewards and distances
        VR = ComputeVirtualReward(e)

        # If rewards where 0., maximise distance (exploration)
        if all(isnan.(VR))
            VR = D
        end

        ### Update agents and run simulation
        for (i,walker) in enumerate(e.walkers)
            j = DifferentIndexInInterval(i, 1:n_walkers)

            if VR[i] == 0.
                Clone!(e, i, j)
            elseif VR[i] > VR[j]
                nothing
            else
                p = CloneProbability(VR[i], VR[j])
                if rand() < p
                    Clone!(e, i, j)
                end
            end
            # TODO: Find a better logging system
            statesHistory = Simulate!(e, walker)
            append!(log[i], statesHistory)
        end
        ScanLoopHook(e)
    end
    log
end

""" Default decision: take most popular action """
function Decide(e::AbstractEnvironment{<:DiscreteModel})
    cmap = countmap(map(x->x.initialDecision, e.walkers))
    argmax(cmap)
end

end # module
