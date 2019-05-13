using NLopt
using Interpolations
using Statistics
using NLsolve
using Debugger
using Roots
#define parameters
α = 0.4
β = 0.99
η = 1.5
δ = 0.025

function SScondition!(F,X)
    Y,C,N,I,K = [X[i] for i=1:5]
    F[1] = η/(1-N) - 1/C*(1-α)*Y/N
    F[2] = Y - N^(1-α)*(K^α)
    F[3] = C + I - Y
    F[4] = (1-δ)*K+I-K
    F[5] = β*(α*Y/K+1-δ)-1
end
initial_ss = [0.1 0.1 0.1 0.1 0.1]
# solve the steadystate
ss = nlsolve(SScondition!,initial_ss,autodiff =:forward).zero
Y_ss,C_ss,N_ss,I_ss,K_ss = [ss[i] for i=1:5]


# get the grid for k
#grid for K
nk = 200;
dev = 0.2;
kmin = (1-dev)*K_ss
kmax = (1+dev)*K_ss;
kstep = (kmax-kmin)/(nk-1);
grid_k = kmin:kstep:kmax;

# change iterator to loops
function T(w, grid, β, u, f, shocks, Tw = similar(w);compute_policy = false)
    w_func = LinearInterpolation(grid, w)
    if compute_policy
        σ = similar(w)
    end
    function labor!(k_prime)
        k= kgrid[1]
        function labor_closure(k_prime,k)
            η,δ,β,α = 1.5, 0.025, 0.99, 0.4;
            f(N) = η/(1-N)*(N^α) - 1/(k^α*N^(1-α)-k_prime + (1-δ)*k)*(1-α)*((k^α));
            return find_zero(f,[0,10])
        end
    return labor_closure(k_prime,k)
    end

    for (i,k) in enumerate(grid_k)
        objectives = (k_prime,l) -> -1.0*(u(f(k,l) - k_prime + (1-δ)k,l) + β*mean(w_func.(k).*shocks) where l = labor!(k_prime))

        opt = Opt(:LN_COBYLA, 2)
        opt.lower_bounds = [0.0, 0.0]
        opt.upper_bounds = [40.0,1.0]
        opt.xtol_rel = 1e-4

        opt.min_objective = objectives
        (minf,minx,ret) = NLopt.optimize(opt, [0.2, 0.3])
        #@show results
        Tw = -1.0*minf
        @show Tw
        @show minx
        @show ret
    end
end

# define the shocks
using Random
Random.seed!(42) # For reproducible results.
μ = 1
s = 0.25
shock_size = 250     # Number of shock draws in Monte Carlo integral

# grid_y = range(1e-5,  grid_max, length = grid_size)
shocks = exp.(μ .+ s * randn(shock_size))

w1 = log.(grid_k);  # An initial condition -- fairly arbitrary
n = 35
utility(c,l) = log(c)+η*log(1-l)
production(k,l) = (k^α)*l^(1-α);


w = T(w1,grid_k,β,utility, production, shocks)
