a = [1 2 3];

f(x) = x;
function bisection(init,lb,ub)
    ϵ=0.0000000000001
    if ub-lb<ϵ*(1+abs(ub)+abs(lb))
        mid = ub
        println("$mid is right one.")
    else
        mid = (lb+ub)/2
        if f(mid)*f(ub)<0
            ub = mid
        else
            lb = mid;
        end
    end
    return mid
end

function newton(init,ϵ,δ)
    k=0
    x = x - f(x)/
end

# secant method, evaluate the derivatives
# $f'(x)=f(x+h)-f(x)/h$
# x_k+1 = x_k - f(x_k)(x_k-x_k-1)/(f())
