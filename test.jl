"""
using Optim
using Interpolations
using Statistics
using NLsolve
using Debugger

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
dev = 0.8;
kmin = (1-dev)*K_ss
kmax = (1+dev)*K_ss;
#kstep = (kmax-kmin)/(nk-1);
grid_k = range(kmin,kmax,length=nk);
@show grid_k

function T(w, grid, β, u, f, shocks, Tw = similar(w);
                          compute_policy = false)
    w_func = LinearInterpolation(grid, w)
    # objective for each grid point
    objectives = ((k_prime,l) -> (-1).*mean(u(f(k,l).*shocks - k_prime + (1-δ)k,l) + β*w_func.(k).*shocks) for k in grid_k)
    #@show objectives
    lower = [1e-4,1e-4];
    upper = [grid_k,range(1,1,length=nk)];
    initial_x = fill([1e-2,0.002],200)
    # d_obj = OnceDifferentiable.(objectives,initial_x; autodiff= :forward)
    # @show d_obj
    function d_obj!(G,k_prime,l)
        G[1] = (-1)/mean(shocks.*(k^α*l^(1-α)) .- k_prime .+ (1-δ)*k)+β*mean(shocks)
        G[2] =
    results = optimize.(d_obj,lower,upper,initial_x, Fminbox(LBFGS())) # solver result for each grid point
    #@show results
    Tw = Optim.minimum.(results)
    if compute_policy
        σ = Optim.minimizer.(results)
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

w = grid_k .+ 0.5;  # An initial condition -- fairly arbitrary
n = 35
utility(c,l) = log(c)+η*log(1-l)
production(k,l) = (k^α)*l^(1-α);
w = T(w,grid_k,β,utility, production, shocks)

# rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
# results = optimize(rosenbrock, lower, upper, initial_x, Fminbox(NelderMead()))
@enter T(w,grid_k,β,utility, production, shocks)
"""


using NLsolve
using Gadfly

# initialize the parameters
η,δ,β,α = 1.5, 0.025, 0.99, 0.4;

# function for calculating steadystate
function SScondition!(F,X)
    Y,C,N,I,K = [X[i] for i=1:5]
    F[1] = η/(1-N) - 1/C*(1-α)*Y/N
    F[2] = Y - N^(1-α)*(K^α)
    F[3] = C + I - Y
    F[4] = (1-δ)*K+I-K
    F[5] = β*(α*Y/K+1-δ)-1
end
initial_x = [0.1 0.1 0.1 0.1 0.1]
# solve the steadystate
ss = nlsolve(SScondition!,initial_x,autodiff =:forward).zero
Y_ss,C_ss,N_ss,I_ss,K_ss = [ss[i] for i=1:5]

using Roots
# # compute the labor supply decision in each t
#
# function labor!(F,N)
#     C, K = 0.5, 1.0
#     η,δ,β,α = 1.5, 0.025, 0.99, 0.4;
#     Z = 1.5
#     F[1] = η/(1-N)*(N^α) - 1/C*(1-α)*(Z*(K^α))
#     # combine equation (1) and (2) and production function
#     # we have (n_t^α)×η/(1-n_t) = (Z_t×K_t^α/C_t)×(1-α)
# end
#
# function dlabor!(J,N)
#     J[1] = α*(N^(α-1))*η/(1-N) - (1-N)^(-2)*N^α;
# end


# test roots to find the solution to l.

# f(N) = 1.5/(1-N)*(N^0.4) - 1/0.5*(1-0.4)*(1.5*(1.0^0.4))
# find_zero(f,0)
#
# # so better to use Roots to solve one dimensional one equation case
# dlabor!(N) = 0.4*(N^(0.4-1))*1.5/(1-N) - (1-N)^(-2)*1.5^0.4;
# N = nlsolve(labor!,dlabor!,[.2]).zero

# # add labor grids
# nl = 200;
# dev = 0.2;
# lmin = (1-dev)*N_ss
# lmax = (1+dev)*N_ss;
# lstep = (lmax-lmin)/(nl-1);
# lgrid = collect(lmin:lstep:lmax);

#
#
# global v=v0=ev=c0 = zeros(nk,nz); # c0 total wealth
# global c = zeros(nk,1);
#
# global N = ones(nk,1)
# '''
# '''
# v=v0=ev=c0 = zeros(nk,nz); # c0 total wealth
# N = ones(nk,1)
# for iz =1:2
#     c0[:,iz] = zgrid[iz].*(kgrid.^α).*(N.^(1-α)) + (1-δ).*kgrid;
# end
#
# utility function
# uf(CN::Tuple) = 1*(log(CN[1])+(1-η)log(1-CN[2]));# CN is (C, N) a tuple.
using Statistics, Random
using NLsolve
using Gadfly

Random.seed!(42) # For reproducible results.
let
    # grid for productivity z
    nz = 200;
    μ = 1;
    s = 0.1;
    zgrid = exp.(μ .+ s * randn(nz))

    # zgrid = [0.5,1.5];
    # tran_z = [0.8 0.2; 0.2 0.8];
    #grid for K
    nk = 200;
    dev = 0.6;
    kmin = (1-dev)*K_ss
    kmax = (1+dev)*K_ss;
    kstep = (kmax-kmin)/(nk-1);
    kgrid = collect(kmin:kstep:kmax);
    tol = 1e-4; # tolerance for V
    cnt = 1; # iteration counter
    dif = 1
    v = zeros(nk,nz); # c0 total wealth
    v0 = zeros(nk,nz)
    ev = zeros(nk,nz)
    c0 = zeros(nk,nz)
    @show v0[1]
    c = zeros(nk,1);
    N = ones(nk,1)
    c_argmax = zeros(nk,nz)
    l_argmax = zeros(nk,nz)
    final_v = zeros(nk,nz)
    for iz =1:2
        c0[:,iz] = zgrid[iz].*(kgrid.^α).*(N.^(1-α)) + (1-δ).*kgrid;
    end
    uf(CN::Tuple) = 1*(log(CN[1])+(1-η)log(1-CN[2]));# CN is (C, N) a tuple.
while true
    # utility function
    for iz=1:nz
        for ik = 1:nk;
            # @show ik,iz
            c = ones(size(kgrid)).*c0[ik,iz]-kgrid;
            ind = c .> 0
            c = c[ind]
            # @show c
            # function labor(c)
            function labor!(C)
                K,Z = kgrid[ik],zgrid[iz]
                function labor_closure(C,K,Z)
                    η,δ,β,α = 1.5, 0.025, 0.99, 0.4;
                    f(N) = η/(1-N)*(N^α) - 1/C*(1-α)*(Z*(K^α));
                    return find_zero(f,[0,1])
                end
            return labor_closure(C,K,Z)
            end
            l = labor!.(c)
            l_ind = 1 .> l .> 0
            @show l_ind
            #@show l
            #@show size(c) size(l)
            # l = lgrid;
            # ind_l = 1 .> l .>0
            CN = [cn for cn in zip(c,l)]
            # function labor_condition!(N)
            #     labor!(F,Z=zgrid[iz],K=kgrid,η=η,α=α,C=c)
            # end
            # N_init=1
            # N = nlsolve(labor_condition!,N_init,autodiff = :forward).zero
            #v[ik,iz] = max(uf.(c[ind],l)+ev[ind,iz]');
            # v[ik,iz] = findmax(uf.(CN) + repeat(ev[ind,iz],1,size(CN)[2]))[1]
            v[ik,iz] = findmax(uf.(CN) + ev[ind,iz])[1]
            optim_loc = findmax(uf.(CN) + ev[ind,iz])[2]
            @show optim_loc
            c_argmax[ik,iz] = c[optim_loc]
            l_argmax[ik,iz] = l[l_ind][optim_loc]
        end
    end
    ev = β.*mean(v,dims=2);
    @show ev[1]
    # check convegence
    dif = findmax(abs.(v-v0))[1];
    @show dif==0
    v0 = v;
    cnt = cnt +1
    @show cnt
    if dif < tol
        final_v = v
        break
    else
        continue
    end
    @show dif
end

p = Gadfly.plot(x=kgrid, y=final_v[:,1], Geom.line)
img = SVG("yaoming_q3.svg", 6inch, 4inch)
draw(img, p)
p2 = Gadfly.plot(x=kgrid,y=c_argmax[:,1],Geom.line)
img2 = SVG("policy_c_yaoming_q3.svg",6inch,4inch)
draw(img2,p2)

p3 = Gadfly.plot(x=kgrid,y=l_argmax[:,1],Geom.line)
img3 = SVG("policy_l_yaoming_q3.svg",6inch,4inch)
draw(img3,p3)
end


# 5th May 2019
## test the optimization over a function with multiple parameters.
using PyCall
using Statistics
interp = pyimport("scipy.interpolate")
so = pyimport("scipy.optimize")
using Random
Random.seed!(42) # For reproducible results.
let
    # grid for productivity z
    nz = 2;
    μ = 1;
    s = 0.25;
    zgrid = μ .+ s.* randn(nz)
    @show zgrid
    # zgrid = [0.5,1.5];
    # tran_z = [0.8 0.2; 0.2 0.8];
    #grid for K
    nk = 200;
    dev = 0.6;
    kmin = (1-dev)*K_ss
    kmax = (1+dev)*K_ss;
    kstep = (kmax-kmin)/(nk-1);
    kgrid = collect(kmin:kstep:kmax);
    tol = 1e-4; # tolerance for V
    cnt = 1; # iteration counter
    dif = 1
    v = zeros(nk,nz); # c0 total wealth
    v0 = zeros(nk,nz)
    ev_func(k,z) = (2k+1)+1;
    v_func(k,z) = 2k +1 +z;
    c0 = zeros(nk,nz)
    # @show v0[1]
    c = zeros(nk,1);
    N = 0.5 .* ones(nk,1)
    c_argmax = zeros(nk,nz)
    l_argmax = zeros(nk,nz)
    final_v = zeros(nk,nz)
    for iz =1:nz
        c0[:,iz] = zgrid[iz].*(kgrid.^α).*(N.^(1-α)) + (1-δ).*kgrid;
    end
    uf(CN::Tuple) = 1*(log(CN[1])+(1-η)log(1-CN[2]));# CN is (C, N) a tuple.
while true
    # utility function
    for iz=1:nz
        for ik = 1:nk;
            # @show ik,iz
            c = ones(size(kgrid)).*c0[ik,iz]-kgrid;
            ind = c .> 0
            c = c[ind]
            @show c
            # function labor(c)
            function labor!(C)
                K,Z = kgrid[ik],zgrid[iz]
                function labor_closure(C,K,Z)
                    η,δ,β,α = 1.5, 0.025, 0.99, 0.4;
                    f(N) = η/(1-N)*(N^α) - 1/C*(1-α)*(Z*(K^α));
                    return find_zero(f,[0,1])
                end
            return labor_closure(C,K,Z)
            end
            l = labor!.(c)
            l_ind = 1 .> l .> 0
            @show l_ind
            CN = [cn for cn in zip(c,l)]
            # @show uf.(CN)
            # @show kgrid[ik],zgrid[iz]
            @show v_func(kgrid[ik],zgrid[iz])
            w =v_func(kgrid[ik],zgrid[iz]) .+ uf.(CN)

            @show iz
            @show typeof(w)
            w = w*(-1)
            # iterpolate w over c.
            w_func = interp.interp1d(c,w,fill_value="extrapolate")
            w_func_wrapper(x) = w_func(x)
            x0 = CN[1][1]
            @show findmin(c)
            res = so.minimize_scalar(w_func,bounds=(findmin(c)[1],findmax(c)[1]),method="bounded")
            v[ik,iz] = (-1)*res["fun"]
            # @show res
            # optim_loc = findmax(uf.(CN) + ev[ind,iz])[2]
            c_argmax[ik,iz] = res["x"]
            # @show c_argmax
            l_argmax[ik,iz] = labor!(c_argmax[ik,iz])
        end
    end
    ev_mat = β.*mean(v,dims=2);
    @show ndims(ev_mat),size(kgrid)
    ev_mat = dropdims(ev_mat;dims=2)
    @show size(v)
    ev_func = interp.interp1d(kgrid,ev_mat,fill_value="extrapolate")
    # @show size(v),size(dropdims(kgrid[:,1];dims=2)),size(dropdims(zgrid[:,1];dims=2))
    v_func = interp.interp2d(kgrid[:,1],zgrid[:,1],v)
    @show ev_func(1)
    # @show ev[1]
    # check convegence
    dif = findmax(abs.(v-v0))[1];
    # @show dif==0
    v0 = v;
    cnt = cnt +1
    # @show cnt
    if dif < tol
        final_v = v
        break
    else
        continue
    end
    # @show dif
end
 p = Gadfly.plot(x=kgrid, y=final_v[:,1], Geom.line)
 img = SVG("value_q3.svg", 6inch, 4inch)
 draw(img, p)
 @show c_argmax
 p2 = Gadfly.plot(x=kgrid,y=c_argmax[:,1],Geom.line)
 img2 = SVG("policy_c_q3.svg",6inch,4inch)
 draw(img2,p2)

 p3 = Gadfly.plot(x=kgrid,y=l_argmax[:,1],Geom.line)
 img3 = SVG("policy_l_q3.svg",6inch,4inch)
 draw(img3,p3)
end


function y(k)
    f(l) = (1-0.4)*1*20^0.4*l^(1-0.4)/((1-0.025)*20+20^0.4-k)- 1.5/(1-l);
    lower = [0]
    upper = [1]
    initial_x = [0.3]
    inner_optimizer = GradientDescent()
    results = optimize(f, lower, upper, initial_x, Fminbox(inner_optimizer);autodiff= :forward)

    ## following up plan: follow the logic in working_on.jl and change the max u(c,l) part using
    ## Interpolations, try use pycall to use scipy. First, calculate max over u(c,l) over each grid,
    ## then, do the interpolation over k,z to v(k,z), then find the max ? Or should i interpolate
    ## over (c,l) to objective(c,l), then interpolate to find the max (c*,l*) esstentially it's same
    ## to find c* alone?

    # Plan: to calculate the W(c,l)=u(c,l) + ev using ev = tran*v where v(k,z) = (2k+1)*z. then interp2d(c,l) -> W(c,l), then so.minimize(-1*(W,x0)
function labor!(C)
    K,Z = 20,1
    function labor_closure(C,K,Z)
        η,δ,β,α = 1.5, 0.025, 0.99, 0.4;
        f(N) = η/(1-N)*(N^α) - 1/C*(1-α)*(Z*(K^α));
        return find_zero(f,[0,1])
    end
return labor_closure(C,K,Z)
end
