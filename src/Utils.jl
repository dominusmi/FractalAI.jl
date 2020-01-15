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

""" Calculates the virtual reward probability """
CloneProbability(VR_src::AbstractFloat, VR_dest::AbstractFloat) = (VR_dest-VR_src) / VR_src

""" Overwrite ith walker with jth """
function Clone!(e::AbstractEnvironment, i::Int, j::Int)
    copy!(e.walkers[i], e.walkers[j])
end
