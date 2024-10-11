from fenics import *
import matplotlib.pyplot as plt

# Create mesh and define function space
mesh = UnitIntervalMesh(50)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('sin(pi*x[0])', degree=2)
bc = DirichletBC(V, u_D, 'on_boundary')

# Define initial value
u_n = interpolate(u_D, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
alpha = 1.0
dt = 0.001
f = Constant(0)
F = u*v*dx + dt*alpha*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Time-stepping
u = Function(V)
t = 0
T = 0.5
while t < T:
    t += dt
    solve(a == L, u, bc)
    u_n.assign(u)

# Plot solution
plot(u)
plt.show()
