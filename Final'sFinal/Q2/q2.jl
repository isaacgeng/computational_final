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
----
using Roots
# use let keyword in Julia to define variables working only inside the let/end code chunk.
# Because use a = 10; while a>1 a = a -1 end; would get error "a not defined in the while loop".
let
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
    #---------------------------------------#
    tol = 1e-4; # tolerance for V
    cnt = 1; # iteration counter
    dif = 1
    v = zeros(nk,nz); # c0 total wealth
    v0 = zeros(nk,nz)
    ev = zeros(nk,nz)
    c0 = zeros(nk,nz)
    @show v0[1]
    c = zeros(nk,1);
    N = 0.5*ones(nk,1)
    c_argmax = zeros(nk,nz)
    l_argmax = zeros(nk,nz)
    final_v = zeros(nk,nz)
    for iz =1:2
        c0[:,iz] = zgrid[iz].*(kgrid.^α).*(N.^(1-α)) + (1-δ).*kgrid;
    end
    uf(CN::Tuple) = 1*(log(CN[1])+(η)*log(1-CN[2]));# CN is (C, N) a tuple.
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
            CN = [cn for cn in zip(c,l)]
            v[ik,iz] = findmax(uf.(CN) + ev[ind,iz])[1]
            optim_loc = findmax(uf.(CN) + ev[ind,iz])[2]
            @show optim_loc
            c_argmax[ik,iz] = c[optim_loc]
            l_argmax[ik,iz] = l[l_ind][optim_loc]
        end
    end
    ev = β.*v*tran_z';
    @show ev
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
img = SVG("yaoming.svg", 6inch, 4inch)
draw(img, p)
p2 = Gadfly.plot(x=kgrid,y=c_argmax[:,1],Geom.line)
img2 = SVG("policy_c_yaoming.svg",6inch,4inch)
draw(img2,p2)

p3 = Gadfly.plot(x=kgrid,y=l_argmax[:,1],Geom.line)
img3 = SVG("policy_l_yaoming.svg",6inch,4inch)
draw(img3,p3)
end


#plot(kgrid,final_v[1])
#
# @show dif cnt
#
#
# x = range(0; stop=2*pi, length=1000); y = sin.(3 * x + 4 * cos.(2 * x));
# plot(x, y, color="red", linewidth=2.0, linestyle="--")
# title("A sinusoidally modulated sinusoid")
#
#
# # test forwardiff see if anything goes wrong with nlsolve.
#
# using ForwardDiff
# function f(N)
#     C, K = 0.5, 1.0
#     η,δ,β,α = 1.5, 0.025, 0.99, 0.4;
#     Z = 1.5
#     f = η/(1-N)*(N.^α) - 1/C*(1-α)*(Z*(K.^α))
#     return f
# end
#
# g = N -> ForwardDiff.gradient(f,N)
# @show g
# g()
#
# function labor!(C)
#     K,Z = 16,0.5
#     function labor_closure(C,K,Z)
#         η,δ,β,α = 1.5, 0.025, 0.99, 0.4;
#         f(N) = η/(1-N)*(N^α) - 1/C*(1-α)*(Z*(K^α));
#         return find_zero(f,3)
#     end
# return labor_closure(C,K,Z)
# end
# (3) ---------------------------------------------------#
# use let keyword in Julia to define variables working only inside the let/end code chunk.
# Because use a = 10; while a>1 a = a -1 end; would get error "a not defined in the while loop".
let
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
    #---------------------------------------#
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
    @show c0
    uf(CN::Tuple) = 1*(log(CN[1])+η*log(1-CN[2]));# CN is (C, N) a tuple.
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
            CN = [cn for cn in zip(c,l)]
            v[ik,iz] = findmax(uf.(CN) + ev[ind,iz])[1]
            optim_loc = findmax(uf.(CN) + ev[ind,iz])[2]
            @show optim_loc
            c_argmax[ik,iz] = c[optim_loc]
            l_argmax[ik,iz] = l[l_ind][optim_loc]
        end
    end
    ev = β.*v*tran_z';
    @show ev
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
img = SVG("yaoming.svg", 6inch, 4inch)
draw(img, p)
p2 = Gadfly.plot(x=kgrid,y=c_argmax[:,1],Geom.line)
img2 = SVG("policy_c_yaoming.svg",6inch,4inch)
draw(img2,p2)

p3 = Gadfly.plot(x=kgrid,y=l_argmax[:,1],Geom.line)
img3 = SVG("policy_l_yaoming.svg",6inch,4inch)
draw(img3,p3)
end
