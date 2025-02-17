import numpy as np
import time

# consts
h = 0.025
g = 9.81

# constraints
l1 = 1
l2 = 1
m1 = 1
m2 = 1

def polar_to_cart(l, θ):
    x = l * np.sin(θ)
    y = l * np.cos(θ)
    return x, y 

def fθ1(t, θ1, θ2, ω1, ω2):
    return ω1

def fθ2(t, θ1, θ2, ω1, ω2):
    return ω2

def fω1(t, θ1, θ2, ω1, ω2):
    Δθ = θ1 - θ2

    dividend = (
        -g*(2*m1+m2)*np.sin(θ1) 
        -m2*g*np.sin(θ1-2*θ2) 
        -2*np.sin(Δθ)*m2*(l2*ω2**2 + l1*np.cos(Δθ)*ω1**2)
    )

    divisor = (
        l1*(2*m1 + m2 - m2*np.cos(2*Δθ))
    )

    return dividend / divisor

def fω2(t, θ1, θ2, ω1, ω2):
    Δθ = θ1 - θ2

    dividend = (
        2*np.sin(Δθ)*(
            (m1+m2)*l1*ω1**2 + 
            g*(m1+m2)*np.cos(θ1) + 
            m2*l2*np.cos(Δθ)*ω2**2
        )
    )

    divisor = (
        l2*(2*m1 + m2 - m2*np.cos(2*Δθ))
    )

    return dividend / divisor

def RK4(t, h, θ1, θ2, ω1, ω2):
    k1θ1 = fθ1(t, θ1, θ2, ω1, ω2)
    k1θ2 = fθ2(t, θ1, θ2, ω1, ω2)
    k1ω1 = fω1(t, θ1, θ2, ω1, ω2)
    k1ω2 = fω2(t, θ1, θ2, ω1, ω2)

    k2θ1 = fθ1(t + h/2, θ1 + h*k1θ1/2, θ2 + h*k1θ2/2, ω1 + h*k1ω1/2, ω2 + h*k1ω2/2)
    k2θ2 = fθ2(t + h/2, θ1 + h*k1θ1/2, θ2 + h*k1θ2/2, ω1 + h*k1ω1/2, ω2 + h*k1ω2/2)
    k2ω1 = fω1(t + h/2, θ1 + h*k1θ1/2, θ2 + h*k1θ2/2, ω1 + h*k1ω1/2, ω2 + h*k1ω2/2)
    k2ω2 = fω2(t + h/2, θ1 + h*k1θ1/2, θ2 + h*k1θ2/2, ω1 + h*k1ω1/2, ω2 + h*k1ω2/2)
    
    k3θ1 = fθ1(t + h/2, θ1 + h*k2θ1/2, θ2 + h*k2θ2/2, ω1 + h*k2ω1/2, ω2 + h*k2ω2/2)
    k3θ2 = fθ2(t + h/2, θ1 + h*k2θ1/2, θ2 + h*k2θ2/2, ω1 + h*k2ω1/2, ω2 + h*k2ω2/2)
    k3ω1 = fω1(t + h/2, θ1 + h*k2θ1/2, θ2 + h*k2θ2/2, ω1 + h*k2ω1/2, ω2 + h*k2ω2/2)
    k3ω2 = fω2(t + h/2, θ1 + h*k2θ1/2, θ2 + h*k2θ2/2, ω1 + h*k2ω1/2, ω2 + h*k2ω2/2)
    
    k4θ1 = fθ1(t + h, θ1 + h*k3θ1, θ2 + h*k3θ2, ω1 + h*k3ω1, ω2 + h*k3ω2)
    k4θ2 = fθ2(t + h, θ1 + h*k3θ1, θ2 + h*k3θ2, ω1 + h*k3ω1, ω2 + h*k3ω2)
    k4ω1 = fω1(t + h, θ1 + h*k3θ1, θ2 + h*k3θ2, ω1 + h*k3ω1, ω2 + h*k3ω2)
    k4ω2 = fω2(t + h, θ1 + h*k3θ1, θ2 + h*k3θ2, ω1 + h*k3ω1, ω2 + h*k3ω2)

    θ1 = θ1 + h/6 * (k1θ1 + 2*k2θ1 + 2*k3θ1 + k4θ1)
    θ2 = θ2 + h/6 * (k1θ2 + 2*k2θ2 + 2*k3θ2 + k4θ2)
    ω1 = ω1 + h/6 * (k1ω1 + 2*k2ω1 + 2*k3ω1 + k4ω1)
    ω2 = ω2 + h/6 * (k1ω2 + 2*k2ω2 + 2*k3ω2 + k4ω2)

    return θ1, θ2, ω1, ω2

def calculate():
    # variables
    θ1 = 0.785
    θ2 = 2.5
    ω1 = 0.785
    ω2 = 1.572

    t = 0
    while(True):
        θ1, θ2, ω1, ω2 = RK4(t, h, θ1, θ2, ω1, ω2)
        t += h
        yield θ1, θ2, ω1, ω2

def main():

    for θ1, θ2, ω1, ω2 in calculate():
        #x1, y1, x2, y2 = polar_to_cart(θ1, θ2)
        #print(f"({x1:.2f}, {x2:.2f})\r", end="")
        print(f"({θ1:0.3f}, {ω1:0.3f}) ({θ2:0.3f}, {ω2:0.3f})    \r", end = "")
        time.sleep(0.1)

if __name__ == "__main__":
    main()