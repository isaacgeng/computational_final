using Optim
using Statistics
function T(w, grid, β, u, f, shocks, Tw = similar(w);
                          compute_policy = false)
    w_func = LinearInterpolation(grid, w)
    # objective for each grid point
    objectives = (c -> u(c) + β * mean(w_func.(f(y - c) .* shocks)) for y in grid_y)
    @show typeof(objectives)
    results = maximize.(objectives, 1e-10, grid_y) # solver result for each grid point
    Tw = Optim.maximum.(results)
    @show Tw
    if compute_policy
        σ = Optim.maximizer.(results)
        return Tw, σ
    end
    return Tw
end

using Random
Random.seed!(42) # For reproducible results.

grid_max = 4         # Largest grid point
grid_size = 200      # Number of grid points
shock_size = 250     # Number of shock draws in Monte Carlo integral

grid_y = range(1e-5,  grid_max, length = grid_size)
shocks = exp.(0 .+ 0.1 * randn(shock_size))

w = 5 * log.(grid_y)  # An initial condition -- fairly arbitrary
n = 35

for i in 1:n
    w = 5 * log.(grid_y)  # An initial condition -- fairly arbitrary
    w = T(w, grid_y, β, log, k -> k^α, shocks)
#    plot!(grid_y, w, color = RGBA(i/n, 0, 1 - i/n, 0.8), linewidth = 2, alpha = 0.6,
#          label = "")
end
