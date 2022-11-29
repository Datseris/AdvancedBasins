using DrWatson
@quickactivate
using OrdinaryDiffEq:Vern9
using DynamicalSystemsBase
using Attractors
using CairoMakie
using LinearAlgebra

function glv!(du, u, p, t)
    @unpack A, r = p
    dN = view(du, :)
    N = view(u, :)
    mul!(dN, A, N)
    @. dN = dN + r
    @. dN = N * dN
    # @. dN = log(dN)
    nothing
end

A = [−7.43e−07 3.23e−04 −7.21e−05 −1.69e−07;
    −9.92e−08 −9.14e−07 −1.43e−05 4.20e−07;
    −1.13e−07 −4.6e−05 −5.48e−07 −5.09e−05;
    −2.11e−07 −1.99e−06 −6.72e−06 −7.08e−08]

Alog = log.([−7.43e−07 3.23e−04 −7.21e−05 −1.69e−07;
    −9.92e−08 −9.14e−07 −1.43e−05 4.20e−07;
    −1.13e−07 −4.6e−05 −5.48e−07 −5.09e−05;
    −2.11e−07 −1.99e−06 −6.72e−06 −7.08e−08])
r = [ 6.89 6.65 7.69 6.17 ]'
p = @strdict A r
u0s = rand(Float64, 4)
ds = ContinuousDynamicalSystem(glv!, u0s, p)

u0 = rand(4);
T = 1000; Ttr = 0.0; Δt = 0.1;
tr = trajectory(ds, T; Ttr, Δt); t = Ttr:Δt:T;

fig = Figure(res=(1000, 800))
for i=1:4
    ax = Axis(fig[i, 1])
    lines!(ax, t, tr[:,i])
end
# ax = Axis3(fig[2:4,1:3])
# lines!(ax, tr[:,1], tr[:,2], tr[:,3])
fig

#find all attractors
diffeq = (reltol = 1e-9,  alg = Vern9());
xg = yg = zg = wg = range(0, 1e8,length = 20);
mapper = AttractorsViaRecurrences(ds, (xg, yg, zg, wg);
        # mx_chk_fnd_att = 1000,
        # mx_chk_loc_att = 500,
        # mx_chk_att = 100,
         sparse = true,
        # Ttr = 100,
        # Δt=0.1,
        diffeq,
        stop_at_Δt=true,
         );
xg = yg = zg = wg = range(0, 1e8,length=20);
basins, atts = basins_of_attraction(mapper, (xg, yg, zg, wg));
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
save("$(plotsdir())/glv.png", fig)