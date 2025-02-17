import pygame as pg
import time
import engine

def main():
    WIDTH, HEIGHT = 600, 600
    window = pg.display.set_mode((WIDTH, HEIGHT))
    
    for θ1, θ2, ω1, ω2 in engine.calculate():
        window.fill((255, 255, 255))

        x1, y1 =  engine.polar_to_cart(engine.l1, θ1) 
        x2, y2 =  engine.polar_to_cart(engine.l2, θ2)
        x2 += x1
        y2 += y1

        x1 = (x1 * WIDTH//4) // 1 + WIDTH//2
        y1 = (y1 * HEIGHT//4) // 1 + HEIGHT//2
        x2 = (x2 * WIDTH//4) // 1 + WIDTH//2
        y2 = (y2 * HEIGHT//4) // 1 + HEIGHT//2

        pg.draw.line(window, (0, 0, 0), (WIDTH//2, HEIGHT//2), (x1, y1), 10)
        pg.draw.line(window, (0, 0, 0), (x1, y1), (x2, y2), 10)
        
        pg.display.update()

        time.sleep(0.1)

main()