using DrWatson
@quickactivate

using DynamicalSystems
pitchfork(u, p, t) =  SVector{1}(p[1]*u[1] - u[1]^3)
jac(u, p, t) = SVector{1}(p[1] - 3*u[1]^2)
r = [0.5]
u0 = -0.4
ds = ContinuousDynamicalSystem(pitchfork, [u0], r, jac)
xg = range(-3.0, 3.0, length = 10)
mapper = AttractorsViaRecurrences(ds, (xg,))
basins, atts = basins_of_attraction(mapper, (xg,))
basins