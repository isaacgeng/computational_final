using NLsolve
using Optim
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
# compute the labor supply decision in each t
function labor!(F,N)
    C, K = 0.5, 1.0
    η,δ,β,α = 1.5, 0.025, 0.99, 0.4;
    Z = 1.5
    F[1] = η/(1-N)*(N^α) - 1/C*(1-α)*(Z*(K^α))
    # combine equation (1) and (2) and production function
    # we have (n_t^α)×η/(1-n_t) = (Z_t×K_t^α/C_t)×(1-α)
end
function dlabor!(J,N)
    J[1] = α*(N^(α-1))*η/(1-N) - (1-N)^(-2)*N^α;
end
f(N) = 1.5/(1-N)*(N^0.4) - 1/0.5*(1-0.4)*(1.5*(1.0^0.4))
find_zero(f,0)
# so better to use Roots to solve one dimensional one equation case
dlabor!(N) = 0.4*(N^(0.4-1))*1.5/(1-N) - (1-N)^(-2)*1.5^0.4;

@enter dlabor!(0.2)
N = nlsolve(labor!,dlabor!,[.2]).zero

# grid for productivity z
nz = 2;
zgrid = [0.5,1.5];
tran_z = [0.8 0.2; 0.2 0.8];
#grid for K
nk = 200;
dev = 0.2;
kmin = (1-dev)*K_ss
kmax = (1+dev)*K_ss;
kstep = (kmax-kmin)/(nk-1);
kgrid = collect(kmin:kstep:kmax);
# add labor grids
nl = 200;
dev = 0.2;
lmin = (1-dev)*N_ss
lmax = (1+dev)*N_ss;
lstep = (lmax-lmin)/(nl-1);
lgrid = collect(lmin:lstep:lmax);

#

global v=v0=ev=c0 = zeros(nk,nz); # c0 total wealth
global c = zeros(nk,1);

global N = ones(nk,1)

for iz =1:2
    c0[:,iz] = zgrid[iz].*(kgrid.^α).*(N.^(1-α)) + (1-δ).*kgrid;
end

# utility function
uf(CN::Tuple) = 1*(log(CN[1])+(1-η)log(1-CN[2]));# CN is (C, N) a tuple.

tol = 1e-4; # tolerance for V
cnt = 1; # iteration counter
dif = 1
while true

    tol = 1e-4; # tolerance for V
    cnt = 1; # iteration counter
    dif = 1
    v=v0=ev=c0 = zeros(nk,nz); # c0 total wealth
    c = zeros(nk,1);
    for iz =1:2
        c0[:,iz] = zgrid[iz].*(kgrid.^α).*(N.^(1-α)) + (1-δ).*kgrid;
    end

    # utility function
    uf(CN::Tuple) = 1*(log(CN[1])+(1-η)log(1-CN[2]));# CN is (C, N) a tuple.
    for iz=1:nz
        for ik = 1:nk;
            @show ik,iz
            c = ones(size(kgrid)).*c0[ik,iz]-kgrid;
            ind = c .> 0
            l = lgrid;
            ind_l = 1 .> l .>0
            CN = [(c,n) for c in c[ind], n in lgrid[ind_l]]
            #        @show size(CN), size(ev[:,iz])
            # function labor_condition!(N)
            #     labor!(F,Z=zgrid[iz],K=kgrid,η=η,α=α,C=c)
            # end
            # @show zgrid[iz]
            # N_init=1
            # N = nlsolve(labor_condition!,N_init,autodiff = :forward).zero
            #v[ik,iz] = max(uf.(c[ind],l)+ev[ind,iz]');
            #        @show uf.(CN) + repeat(ev[ind,iz],1,size(CN)[2])
            v[ik,iz] = findmax(uf.(CN) + repeat(ev[ind,iz],1,size(CN)[2]))[1]
        end
    end
    ev = β.*v*tran_z';
    # check convegence
    dif = findmax(abs.(v-v0))[1];
    v0 = v;
    cnt = cnt +1
    if dif < tol
        global final_v = v
        break
    end
end

#plot(kgrid,final_v[1])

p = Gadfly.plot(x=kgrid, y=final_v[:,1], Geom.line)
img = SVG("test1.svg", 6inch, 4inch)
draw(img, p)
@ show dif cnt

x = range(0; stop=2*pi, length=1000); y = sin.(3 * x + 4 * cos.(2 * x));
plot(x, y, color="red", linewidth=2.0, linestyle="--")
title("A sinusoidally modulated sinusoid")


using ForwardDiff
function f(N)
    C, K = 0.5, 1.0
    η,δ,β,α = 1.5, 0.025, 0.99, 0.4;
    Z = 1.5
    f = η/(1-N)*(N.^α) - 1/C*(1-α)*(Z*(K.^α))
    return f
end

g = N -> ForwardDiff.gradient(f,N)
@show g
g()
