using NLopt
using Interpolations
using Statistics
using NLsolve
using Debugger
using PyCall
interp = pyimport("scipy.interpolate")
so = pyimport("scipy.optimize")
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


function T(w, grid, β, u, f, shocks, Tw = similar(w);
                          compute_policy = false)
    w_func = CubicSplineInterpolation(grid, w,extrapolation_bc=Reflect())
    # objective for each grid point
    objectives = ((k_prime,l) -> mean(u(f(k,l).*shocks - k_prime + (1-δ)k,l) + β*w_func.(k).*shocks) for k in grid_k)
    @show objectives
    # lower = [0,0];
    # upper = [grid_k,1];
    initial_x = (1e-2,0.002)
    opt = Opt(:LD_MMA, 2)
    # opt.lower_bounds = [0., 0.]
    # opt.upper_bounds = [40,1]
    bounds = [(0,0),(40,1)]
    res = so.minimize((-1)*objectives,initial_x,x0,bounds=bounds)
    @show so.minimize((-1)*objectives,initial_x,x0,bounds=bounds)
    # opt.xtol_rel = 1e-4

    # opt.min_objective = (-1).*objectives
    # (minf,minx,ret) = NLopt.optimize(opt, [1.0, 0.1])
    #@show results
    Tw = (-1)*res.fun
    if compute_policy
        σ = res.x
        return Tw, σ
    end
    return Tw
end

# define the shocks
using Random
Random.seed!(42) # For reproducible results.
μ = 1
s = 0.25
shock_size = 250     # Number of shock draws in Monte Carlo integral

# grid_y = range(1e-5,  grid_max, length = grid_size)
shocks = exp.(μ .+ s * randn(shock_size))

w1 = grid_k .+ 0.5;  # An initial condition -- fairly arbitrary
n = 35
utility(c,l) = log(c)+η*log(1-l)
production(k,l,shocks) = shocks*(k^α)*l^(1-α);
w = T(w1,grid_k,β,utility, production, shocks)

# rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
# results = optimize(rosenbrock, lower, upper, initial_x, Fminbox(NelderMead()))
@enter T(w,grid_k,β,utility, production, shocks)

# change iterator to loops
function T(w, grid, β, u, f, shocks, Tw = similar(w);
    compute_policy = false)
    w_func = interp.interp1d(grid, w)
    if compute_policy
        σ = similar(w)
    end

    for (i,k) in enumerate(grid_k)
        objectives = (k_prime,l) -> u(f(k,l) - k_prime + (1-δ)k,l) + β*mean(w_func.(k).*shocks)

        opt = Opt(:LD_MMA, 2)
        opt.lower_bounds = [0., 0.]
        opt.upper_bounds = [40,1]
        opt.xtol_rel = 1e-4

        opt.min_objective = (-1).*objectives
        (minf,minx,ret) = NLopt.optimize(opt, [1.0, 0.1])
        #@show results
        Tw = (-1)*minf
    end
end
    # objective for each grid point

objectives = ((k_prime,l) -> u(f(k,l) - k_prime + (1-δ)k,l) + β*mean(w_func.(k).*shocks)) for k in grid_k)
@show objectives
lower = [0,0];
upper = [grid_k,1];
initial_x = [1e-2,0.002]

results = maximize.(objectives, lower, upper, initial_x, Fminbox(NelderMead())) # solver result for each grid point
#@show results
Tw = Optim.maximum.(results)
if compute_policy
σ = Optim.maximizer.(results)
return Tw, σ
end
return Tw
end

using Optim
# an example to test.
function obj(x)
    y = 2*x
    z = 4*x
    function obj_closure(x,y,z)
        res = -(x^2 + y^2 + z^2)
    end
    return obj_closure(x,y,z)
end
f(x) = x->obj(x)
res = maximize(f,-1,2)
