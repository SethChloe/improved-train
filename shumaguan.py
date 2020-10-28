import turtle,time
import random as r
def drawGap():
    turtle.penup()
    turtle.fd(5)
def drawline(draw):
    drawGap()
    turtle.pendown() if draw else turtle.penup()
    turtle.fd(40)
    drawGap()
    turtle.right(90)
def drawdigit(digit):
    drawline(True) if digit in [2,3,4,5,6,8,9] else drawline(False)
    drawline(True) if digit in [0,1,3,4,5,6,7,8,9] else drawline(False)
    drawline(True) if digit in [0,2,3,5,6,8,9] else drawline(False)
    drawline(True) if digit in [0,2,6,8] else drawline(False)
    turtle.left(90)
    drawline(True) if digit in [0,4,5,6,8,9] else drawline(False)
    drawline(True) if digit in [0,2,3,5,6,7,8,9] else drawline(False)
    drawline(True) if digit in [0,1,2,3,4,7,8,9] else drawline(False)
    turtle.left(180)
    turtle.penup()
    turtle.fd(20)
def drawdate(date):
    turtle.pencolor('red')
    for i in date:
        if i=='-':
            turtle.write("年",font=("Arial",18,"normal"))
            turtle.pencolor("purple")
            turtle.fd(40)
        elif i=='=':
            turtle.write("月",font=("Arial",18,"normal"))
            turtle.pencolor("blue")
            turtle.fd(40)
        elif i=='+':
            turtle.write("日",font=("Arial",18,"normal"))
        else:
            drawdigit(eval(i))
def main():
    turtle.setup(800, 800, 200, 200)
    turtle.speed(15)
    turtle.penup()
    turtle.fd(-300)
    turtle.pensize(5)
    drawdate(time.strftime("%Y-%m=%d+",time.gmtime()))
main()

def pink():
    color = (1, r.random(), 1)
    return color
def randomrange(min, max):
    return min + (max- min)*r.random()
def moveto(x, y):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()
def heart(r, a):
    factor = 180
    turtle.seth(a)
    turtle.circle(-r, factor)
    turtle.fd(2 * r)
    turtle.right(90)
    turtle.fd(2 * r)
    turtle.circle(-r, factor)

turtle.speed(15)
turtle.pensize(1)
turtle.penup()
for i in range(20):
    turtle.goto(randomrange(-300, 300), randomrange(-300, 300))
    turtle.begin_fill()
    turtle.fillcolor(pink())
    heart(randomrange(10, 50), randomrange(0, 90))
    turtle.end_fill()
moveto(400, -400)
turtle.done()