@kwdef struct floquet_basis
    e_quasi :: Vector
    modes :: Function
    T :: Number
end

function (f::floquet_basis)(t; kwargs...)
    return f.modes(t; kwargs...)
end

function get_floquet_basis(H::Union{QuantumObject, QobjEvo}, T; propagator_kwargs...)
    ti = 0.0
    if :ti in keys(propagator_kwargs)
        ti = propagator_kwargs[:ti]
    end
    U_T = propagator(H, T+ti; propagator_kwargs...)

    eigvals, eigvecs = eigenstates(U_T)
    eigvals = -angle.(eigvals)/T#imag(log.(λs))
    
    return floquet_basis(eigvals, t->propagate_floquet_modes(eigvecs, H, t, T; propagator_kwargs...), T)
end

function propagate_floquet_modes(modes_t0, H, t, T; propagator_kwargs...)
    if t%T == 0
        return modes_t0
    end
    U_t = propagator(H, t%T; propagator_kwargs...)
    return [U_t*modes_t0[i] for i in 1:length(modes_t0)]
end