export EnsembleTimeEvolutionProblem
"""
    EnsembleTimeEvolutionProblem{PT<:TimeEvolutionProblem, PF<:Function}

A structure representing an ensemble time evolution problem for quantum systems.

# Fields
- `prob::PT`: The base time evolution problem.
- `func::PF`: A function used to modify or sample parameters for each trajectory in the ensemble.
- `iterate_params::Bool`: If `true`, parameters are iterated for each trajectory; otherwise, the same parameters are used.
- `full_iterator::AbstractArray`: An array containing all parameter sets or states to be used in the ensemble.
- `n_states::Int`: The number of initial states.
- `trajectories::Int`: The total number of trajectories to simulate.

# Usage
This is used when setting up ensemble sesolve problems, useful for simulating multiple quantum states or parameter sets in parallel. 

Example:
```julia
    H = 2 * π * 0.1 * sigmax()
    ψ0 = basis(2, 0) # spin-up
    tlist = LinRange(0.0, 100.0, 100)

    ψs = [ψ0, basis(2, 1)] # spin-up and spin-down

    params = collect(Iterators.product([0,1,2,3,4,5], [0,1,2,3,4,5], [0,1,2,3,4,5]))
    res = sesolve(H, ψs, tlist; params = params, iterate_params = true, alg = Tsit5(), progress_bar=false);
```
"""
struct EnsembleTimeEvolutionProblem{PT<:TimeEvolutionProblem,PF<:Function}
    prob::PT
    func::PF
    full_iterator::AbstractArray 
    trajectories::Int
end


function sesolve(prob::EnsembleTimeEvolutionProblem, alg::OrdinaryDiffEqAlgorithm = Tsit5(); backend = EnsembleThreads())
   ensemble_prob = EnsembleProblem(prob.prob.prob, prob_func = prob.func)
   sols = solve(ensemble_prob, alg, backend, trajectories = prob.trajectories)
   
   to_return = Array{TimeEvolutionSol}(undef, size(prob.full_iterator))
    for i in 1:length(sols)
        ψt = map(ϕ -> QuantumObject(ϕ, type = Ket(), dims = prob.prob.dimensions), sols[i].u)
        sol = TimeEvolutionSol(
                    prob.prob.times,
                    sols[i].t,
                    ψt,
                    _get_expvals(sols[i], SaveFuncSESolve),
                    sols[i].retcode,
                    sols[i].alg,
                    sols[i].prob.kwargs[:abstol],
                    sols[i].prob.kwargs[:reltol],
                )
        to_return[CartesianIndices(to_return)[i]] = sol
    end
    
    return to_return
end

function sesolve(
    H::Union{AbstractQuantumObject{Operator},Tuple},
    ψ0s::Union{Vector{T}, Tuple{T}},
    tlist::AbstractVector;
    alg::OrdinaryDiffEqAlgorithm = Tsit5(),
    e_ops::Union{Nothing,AbstractVector,Tuple} = nothing,
    params = NullParameters(),
    progress_bar::Union{Val,Bool} = Val(false),
    inplace::Union{Val,Bool} = Val(true),
    iterate_params::Bool = false,
    backend = EnsembleThreads(),
    kwargs...,) where T<:QuantumObject{Ket}

    params_init = params
    if iterate_params
        params_init = params[1]
    end

    prob_init = sesolveProblem(
        H,
        ψ0s[1],
        tlist;
        e_ops = e_ops,
        params = params_init,
        progress_bar = progress_bar,
        inplace = inplace,
        kwargs...,
    )

    if iterate_params
        full_iterator = Iterators.product(ψ0s, params)
    elseif !(params == NullParameters())
        full_iterator = Iterators.product(ψ0s, [params])
    else
        full_iterator = Iterators.product(ψ0s, [NullParameters()])
    end
    full_iterator = collect(full_iterator)
    trajectories = length(full_iterator)

    function ensemble_func(prob, i, repeat)
        p = full_iterator[i][2]
        if (p isa NullParameters) || !(iterate_params)
            return remake(prob, u0 = full_iterator[i][1].data)
        
        else
            return remake(prob, u0 = full_iterator[i][1].data; p = p)
        end
    end
    
    ensemble_prob = EnsembleTimeEvolutionProblem(prob_init, ensemble_func, full_iterator, trajectories)
    
    return sesolve(ensemble_prob, alg; backend = backend)
end