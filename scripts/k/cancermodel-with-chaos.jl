using DrWatson
@quickactivate
using OrdinaryDiffEq:Vern9
using DynamicalSystemsBase
using Attractors
using CairoMakie


function cancermodel(u, p, t)
    x1, x2, x3 = u
    a12, a13, a21, a31, r2, r3, k3, d3 = p
    du1 = x1*(1-x1) - a12*x1*x2 - a13*x1*x3 #tumor cells
    du2 = r2*x2*(1-x2) - a21*x1*x2 #host cells
    du3 = (r3*x1*x3)/(x1+k3) - a31*x1*x3 - d3*x3 #effector cells
    return SVector{3}(du1, du2, du3)
end;

a12 = 1; a13 = 2.5; a21 = 1.5; a31 = 0.2; r2 = 0.6; r3 = 4.5; k3 = 1; d3 = 0.5; #6 equilibria in the positive octant + chaotic attractor

p = [a12, a13, a21, a31, r2, r3, k3, d3];
ds = ContinuousDynamicalSystem(cancermodel, rand(3), p);

#integrate ics to see attractors
u0 = rand(3);
T = 1000; Ttr = 0.0; Δt = 0.1;
tr = trajectory(ds, T; Ttr, Δt); t = Ttr:Δt:T;

fig = Figure(res=(1000, 800))
for i=1:3
    ax = Axis(fig[1, i])
    lines!(ax, t, tr[:,i])
end
ax = Axis3(fig[2:4,1:3])
lines!(ax, tr[:,1], tr[:,2], tr[:,3])
fig
save("$(plotsdir())/cancermodel-example-dynamics.png", fig)

#find all attractors
diffeq = (reltol = 1e-9,  alg = Vern9());
xg = yg = zg = range(0, 5,length = 1000);
mapper = AttractorsViaRecurrences(ds, (xg, yg, zg);
        mx_chk_fnd_att = 1000,
        mx_chk_loc_att = 500,
        # mx_chk_att = 100,
         sparse = true,
        Ttr = 100,
        Δt=0.1,
        diffeq,
        stop_at_Δt=true,
         );
xg = yg = zg = range(0, 1,length=100);
basins, atts = basins_of_attraction(mapper, (xg, yg, zg));
basins
atts[1]
fig = Figure(res=(1000, 1000))
ax = Axis3(fig[1,1])
for (k,att) in atts
    if length(att) == 1
        scatter!(ax, att[:,1], att[:,2], att[:,3], markersize=20)
    else
    lines!(ax, att[:,1], att[:,2], att[:,3])
    end
end
fig
save("$(plotsdir())/cancermodel-attractors.png", fig)