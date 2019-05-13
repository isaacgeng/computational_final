include("value_func_n.jl")

## This file solves the value function iteration 
# clear
# clc

# parameters
α = 0.4;
β  = 0.99;
η   = 1.5;
δ = 0.025;

## solve steady state
# x = [C,K,L]

steady = @(x) [1/x(1)-β*(1+α*x(2)^(α-1)*x(3)^(1-α)-δ)/x(1);...
    η/((1-α)*(1-x(3))*x(2)^α*x(3)^(-α))-1/x(1);...
    δ*x(2)-x(2)^α*x(3)^(1-α)+x(1)];

Y = fsolve(@(x) steady(x),[1,6,0.5]);

cstar = Y(1);
kstar = Y(2);
lstar = Y(3);

## solve value function with Markovian shock

# size of grid
N = 100;

# state
z = [1.5;0.5];

# transition matrix
p = [0.8,0.2;0.2,0.8];

# size of state
n = length(z);

kmin = kstar*0.8;
kmax = kstar*1.2;

kgrid = linspace(kmin,kmax,N); # K
k_prm = zeros(n,N);            # K'

# guess of value function
V0 = ones(n,N); # row 1: z = 1.5 ; row 2: z = 0.5
V  = V0;

# iteration
error   = 1;
tol     = 0.0001;
maxiter = 2000;

# record the error
E1 = zeros(1,maxiter);


# start iteration
iter = 0;

while error > tol && iter < maxiter
    
    # update the value function given guess V0
    for j = 1:N
        for s = 1:n
            k0 = kgrid(j);
            k1 = fminbnd(@(k) value_fun_n(k,k0,V0,α,δ,β,η,z(s),kgrid,p(s,:),n),kmin,kmax);
            
            V(s,j) = - value_fun_n(k1,k0,V0,α,δ,β,η,z(s),kgrid,p(s,:),n);
            k_prm(s,j) = k1;
        end
    end
    
    error = norm(V0-V)
    
    iter = iter + 1
    
    E1(iter) = error;
    
    V0 = V;
end

### plot value function and policy function

L = zeros(n,N);
C = L;

for j = 1:N
    for s = 1:n
        
        # solve h
        f = @(l) (1-α)*z(s)*kgrid(j)^α*l^(-α)/((1-δ)*kgrid(j)+z(s)*kgrid(j)^α*l^(1-α)-k_prm(s,j))-η/(1-l);
        
        L(s,j) = fminbnd(@(l) abs(f(l)),0,1);
        
        # given h, solve c
        C(s,j) = (1-δ)*kgrid(j) + z(s)*kgrid(j)^α*L(s,j)^(1-α) - k_prm(s,j);
    end
end

figure
subplot(2,2,1)
plot(kgrid,V)
legend('z_{H}','z_{L}','Location','southeast')
title('Value Function')

subplot(2,2,2)
plot(kgrid,k_prm)
legend('z_{H}','z_{L}','Location','southeast')
title('Policy Function-Capital')

subplot(2,2,3)
plot(kgrid,C)
legend('z_{H}','z_{L}','Location','southeast')
title('Policy Function-Consumption')

subplot(2,2,4)
plot(kgrid,L)
legend('z_{H}','z_{L}','Location','northeast')
title('Policy Function-Labor')

## solve value function with normal iid shock

# choose grid for z
# discretize z into Nz states, each with occuring prob = 0.01

mu = 1;
sigma = 0.25;

Nz = 10;
zn = zeros(1,Nz);

for i = 1:Nz
    zn(i) = norminv(i/(Nz+1),mu,sigma);
end

# transition matrix
pn = 1/Nz*ones(Nz,Nz);

# guess of value function
V0n = zeros(Nz,N);
Vn  = V0n;

# record the error
E2 = zeros(1,maxiter);

# guess of K'
k_prmn = zeros(Nz,N);

errorn = 1;
iter   = 0;

while tol < errorn && iter < maxiter
    
    # update the value function given guess V0n
    for j = 1:N
        for s = 1:Nz
            k0 = kgrid(j);
            k1 = fminbnd(@(k) value_fun_n(k,k0,V0n,α,δ,β,η,zn(s),kgrid,pn(s,:),Nz),kmin,kmax);
            
            Vn(s,j) = - value_fun_n(k1,k0,V0n,α,δ,β,η,zn(s),kgrid,pn(s,:),Nz);
            k_prmn(s,j) = k1;
        end
    end
    
    errorn = norm(V0n-Vn)
    
    iter = iter + 1
    
    E2(iter) = errorn;
    
    V0n  = Vn;
end

### plot value function and policy function

Ln = zeros(Nz,N);
Cn = Ln;

for j = 1:N
    for s = 1:Nz
        
        # solve h
        f = @(l) (1-α)*zn(s)*kgrid(j)^α*l^(-α)/((1-δ)*kgrid(j)+zn(s)*kgrid(j)^α*l^(1-α)-k_prmn(s,j))-η/(1-l);
        
        Ln(s,j) = fminbnd(@(l) abs(f(l)),0,1);
        
        # given h, solve c
        Cn(s,j) = (1-δ)*kgrid(j) + zn(s)*kgrid(j)^α*Ln(s,j)^(1-α) - k_prmn(s,j);
    end
end

[X,Y] = meshgrid(zn,kgrid);

figure
subplot(2,2,1)
surf(X,Y,Vn')
xlabel('z')
ylabel('K')
zlabel('V')
title('value function')
colorbar

subplot(2,2,2)
surf(X,Y,k_prmn')
xlabel('z')
ylabel('K')
zlabel('K^{\prime}')
title('Policy Function-Capital')
colorbar

subplot(2,2,3)
surf(X,Y,Cn')
xlabel('z')
ylabel('K')
zlabel('C')
title('Policy Function-Consumption')
colorbar

subplot(2,2,4)
surf(X,Y,Ln')
xlabel('z')
ylabel('K')
zlabel('L')
title('Policy Function-Labor')
colorbar

save('result.mat','kgrid','V','k_prm','C','L','Vn','k_prmn','Cn','Ln','cstar','kstar','lstar','E1','E2')
