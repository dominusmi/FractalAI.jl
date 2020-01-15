# FractalAI.jl

FractalAI is an intelligent sampling method based on the process of maximising future entropy. ( arXiv:1803.05049 )

This repository implements a basic working scanning process, for which one simply needs to implement their own model (a basic walker and a step environment is already implemented)

This is simply done by overwriting 4 functions:

```julia
"""
  Your model, used as a descriptor of the dynamics of the system.
"""

"""
  Returns an object representing available actions (e.g. a set, a continuous interval)
  Must return an object on which rand can be called, as rand(Actions(m))
"""
Actions(m::YourModel)
"""
  Used to define the model dynamics. Executes action a on walker w, using model dynamics
"""
ExecuteAction!(m::YourModel, w::AbstractWalker, a::ActionType)

"""
  Run your simulation for dt amount of time. (e.g. in a step environment, this dt would correspond to the number
  of steps to execute).
"""
function Simulate!(e::AbstractEnvironment{YourModel}, w::AbstractWalker, dt::SomeTimeIntervalUnit)

""" Returns the state reward """
Reward(m::YourModel, w::AbstractWalker)

""" Returns the distance between two walkers in the model model """
Distance(m::YourModel, w1::AbstractWalker, w2::AbstractWalker)

```

For a basic example, see examples/GridModel

[Read more about it here](https://arxiv.org/pdf/1803.05049.pdf)


## TODO

- Parallelise
- Find a better logging system
- Implement default decision for continuous model
