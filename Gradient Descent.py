import pandas
import matplotlib.pyplot as plt
import numpy as np

# Cost function
def f(x):
    return x ** 2 + x + 1

# Slope and Derivative
def df(x):
    return 2 * x + 1

 
# Make data
x_1 = np.linspace(start=-3, stop=3, num=100)

# Gradient Descent 
new_x = 3
previous_x = 0
step_multiplier = 0.1
precision = 0.0001

x_list = [new_x]
f_x_list = [f(new_x)]
slope_list = [df(new_x)]


for n in range(500):
    # starting point(is random for example is 3 at the first time)
    previous_x = new_x
    # Cost calculation
    gradient = df(previous_x)
    # Learning (make steep to minimum)
    new_x = previous_x - gradient * step_multiplier

    x_list.append(new_x)
    f_x_list.append(f(new_x))
    slope_list.append(df(new_x))

    step_size = abs(new_x - previous_x)
    
    if step_size < precision:
        print(('this loop repeated for {0} times').format(n))
        break

print("Local minimum is: ", new_x)
print("Slope of df(x) value at this point is :", df(new_x))
print("Cost at this point is: ", f(new_x))


# Draw chart 1
plt.figure(figsize=[12, 5])
# Divide a row to two columns
# Show chart cost function
plt.subplot(1, 3, 1)
plt.ylim(0, 15)
plt.xlim([-3,3])
plt.title('Cost Function')
plt.xlabel('X', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.grid()
plt.plot(x_1, f(x_1))
# Show gradient descent
plt.scatter(x_list, f_x_list, color='green', s=50, alpha=0.7)

# Draw chart 2
# Show chart derivative 
plt.subplot(1, 3, 2)
plt.grid()
plt.ylim(-3, 6)
plt.xlim(-2, 3)
plt.plot(x_1, df(x_1), color='red', linewidth=3)
plt.title('Slope of the Cost function')
plt.xlabel('X', fontsize=16)
plt.ylabel('df(x)', fontsize=16)

# Show slopes in gradient descent
plt.scatter(x_list, slope_list, color='green', s=50, alpha=0.7)

# Draw chart 3 Derivative close up

plt.subplot(1, 3, 3)
plt.grid()
plt.xlim(-0.55, 0.2)
plt.ylim(-0.3, 0.8)
plt.plot(x_1, df(x_1), color='red', linewidth=3)
plt.title('Gradient Descent (close up)')
plt.xlabel('X', fontsize=16)

# Show slopes in gradient descent
plt.scatter(x_list, slope_list, color='green', s=50, alpha=0.7)

plt.show()

