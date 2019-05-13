
using Optim
include("interp1.jl")

function value_fun_n(k,k0,v0,α,δ,β,η,z,kgrid,p,n)
    # solve optimal h given k and k0
    f(l) = (1-α)*z*k0^α*l^(-α)/((1-δ)*k0+z*k0^α*l^(1-α)-k) - η/(1-l);
    
    #h = fminbnd(@(l) abs(f(l)),0,1);
    h = optimize(f,0,1,GoldenSection())
    ## value at k for each state
    g  = zeros(n,1);
    for i = 1:n
        g[i] = interp1(kgrid,v0(i,:),k;method="linear");
    end
    
    ## consumption and new val
    c = (1-δ)*k0 + z*k0^α*h^(1-α) - k;
    if c <= 0
        y = -888888888 - 800*abs(c);
    else
        y = log(c) + η*log(1-h) + β*(p*g);
    end
    y = -y;
end

