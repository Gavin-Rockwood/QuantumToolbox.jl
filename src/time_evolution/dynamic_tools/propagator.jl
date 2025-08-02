export propagator, Propagator

function propagator(H::Union{QuantumObject, QobjEvo}, tf; ti = 0, kwargs...)
    iterating_params = false
    if :iterate_params in keys(kwargs)
        kwargs[:iterate_params] = false
    end
    H_evo = QobjEvo(H)

    if !(:params in keys(kwargs))
        eigvals, ψ0s = eigenstates(H_evo(ti))
    elseif !(kwargs[:params] isa NullParameters)
        eigvals, ψ0s = eigenstates(H_evo(kwargs[:params], ti))
    else 
        eigvals, ψ0s = eigenstates(H_evo(ti))
    end



    tlist = [ti, tf]

    sols = sesolve(H_evo, ψ0s, tlist; kwargs...)
    if iterating_params
        n_params = length(sols[1])
        to_return = Dict{Any, Any}()
        for i in 1:n_params
            param = sols[1][i][2]
            to_return[param] = sols[1][i][1].states[end]*ψ0s[1]'
        end
        for (state, i) in Iterators.product(2:length(ψ0s), 1:n_params)
            param = sols[state][i][2]
            to_return[param] = sols[state][i][1].states[end]*ψ0s[state]'
        end
    else
        to_return = sols[1].states[end]*ψ0s[1]'
        for i in 2:length(ψ0s)
            to_return += sols[i].states[end]*ψ0s[i]'
        end
    end

    return to_return
end

struct Propagator{T}
    H :: T
    eval
end
function Propagator(H::T) where T <:Union{QuantumObject, QobjEvo}
    func = (tf; kwargs...) -> propagator(H, tf; kwargs...)
    Propagator(H, func)
end
function (p::Propagator)(tf; kwargs...)
    return p.eval(tf; kwargs...)
end
function Base.show(io::IO, p::Propagator)
    println(io, "Propagator for Hamiltonian: ", p.H)
    println(io, "Use p(tf) to evaluate the propagator at time tf.")
end
function Base.size(p::Propagator)
    return size(p.H)
end
