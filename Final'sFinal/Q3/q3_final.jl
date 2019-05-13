# 5th May 2019
## test the optimization over a function with multiple parameters.

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
