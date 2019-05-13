using Optim

function T(w, grid, β, u, f, shocks, Tw = similar(w);
                          compute_policy = false)
    w_func = LinearInterpolation(grid, w)
    # objective for each grid point
    objectives = (c -> u(c) + β * mean(w_func.(f(y - c) .* shocks)) for y in grid_y)
    results = maximize.(objectives, 1e-10, grid_y) # solver result for each grid point
    Tw = Optim.maximum.(results)
    if compute_policy
        σ = Optim.maximizer.(results)
        return Tw, σ
    end
    return Tw
end