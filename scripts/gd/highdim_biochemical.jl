using DrWatson
@quickactivate
using DynamicalSystems
using OrdinaryDiffEq, GLMakie

"""
    oval2deterministic()
Generate a dynamical system representing the "Oval2 Epithelial-Mesenchymal Transition model"
from [^Rackauckas2017]. This is a **non-stochastic** version of the model.
It is 19 coupled ODEs which are only stiff during transitions between biological states.

As this is a stiff problem, it is recommended to use
`diffeq = (alg = Rodas5(), abstol = 1e-6)`.

[^Rackauckas2017]:
    Rackauckas, C., & Nie, Q. (2017). Adaptive methods for stochastic differential equations
    via natural embeddings and rejection sampling with memory. Discrete and continuous
    dynamical systems. Series B, 22(7), 2731.
"""
function oval2deterministic()
    #Parameters
    J1_200=3.
    J1_34=0.15
    J2_200=0.2
    J2_34=0.35
    J_2z=0.9
    J_O=0.918
    J_SO=0.5
    # J_ZEB=0.06
    J_ecad1=0.1
    J_ecad2=0.3
    J_ncad1=0.4
    J_ncad2=0.4
    J_ncad3 = 2
    J_snail0=0.6
    J_snail1=1.8
    J_zeb=3.0
    K1=1.0
    K2=1.0
    K3=1.0
    K4=1.0
    K5=1.0
    KTGF=20.
    Ks=100.
    # TGF0=0
    TGF_flg=0.
    Timescale=1000.
    dk_ZR1=0.5
    dk_ZR2=0.5
    dk_ZR3=0.5
    dk_ZR4=0.5
    dk_ZR5=0.5
    k0O=0.35
    k0_200=0.0002
    k0_34=0.001
    k0_snail=0.0005
    k0_zeb=0.003
    kO=1.2
    kOp=10.
    k_200=0.02
    k_34=0.019
    k_OT=1.1
    k_SNAIL=16.
    k_TGF=1.5
    k_ZEB=16.
    k_ecad0=5.
    k_ecad1=15.
    k_ecad2=5.
    k_ncad0=5.
    k_ncad1=2.
    k_ncad2=5.
    k_snail=0.05
    k_tgf=0.05
    k_zeb=0.06
    kdO=1.
    kd_200=0.035
    kd_34=0.035
    kd_SNAIL=1.6
    kd_SR1=0.9
    kd_TGF=0.9
    kd_ZEB=1.66
    kd_ecad=0.05
    kd_ncad=0.05
    kd_snail=0.09
    kd_tgf=0.1
    kd_tgfR=1.0
    kd_zeb=0.1
    kd_Op = 10.
    lamda1=0.5
    lamda2=0.5
    lamda3=0.5
    lamda4=0.5
    lamda5=0.5
    lamdas=0.5
    lamdatgfR=0.8
    nO=6.
    nSO=2.
    nzo=2.
    GE = 1.
    f = function (dy,y,p,t)
      # y(1) = snailt
      # y(2) = SNAIL
      # y(3) = miR34t
      # y(4) = SR1 # abundance of SNAIL/miR34 complex
      # y(5) = zebt
      # y(6) = ZEB
      # y(7) = miR200t
      # y(8) = ZR1 # abundance of ZEB/miR200 complex with i copies of miR200 bound on the sequence of ZEB1
      # y(9) = ZR2
      # y(10) = ZR3
      # y(11) = ZR4
      # y(12) = ZR5
      # y(13) = tgft
      # y(14) = TGF
      # y(15) = tgfR # abundance of TGF/miR200 complex
      # y(16) = Ecad
      # y(17) = Ncad
      # y(18) = Ovol2
      TGF0=.5(t>100)
      #ODEs
      dy[1]=k0_snail+k_snail*(((y[14]+TGF0)/J_snail0))^2/(1+(((y[14]+TGF0)/J_snail0))^2+(y[19]/J_SO)^nSO)/(1+y[2]/J_snail1)-kd_snail*(y[1]-y[4])-kd_SR1*y[4]
      dy[2]=k_SNAIL*(y[1]-y[4])-kd_SNAIL*y[2]
      dy[3]=k0_34+k_34/(1+((y[2]/J1_34))^2+((y[6]/J2_34))^2)-kd_34*(y[3]-y[4])-kd_SR1*y[4]+lamdas*kd_SR1*y[4]
      dy[4]=Timescale*(Ks*(y[1]-y[4])*(y[3]-y[4])-y[4])
      dy[5]=k0_zeb+k_zeb*((y[2]/J_zeb))^2/(1+((y[2]/J_zeb))^2+((y[19]/J_2z))^nO)-kd_zeb*(y[5]-(5*y[8]+10*y[9]+10*y[10]+5*y[11]+y[12]))-dk_ZR1*5*y[8]-dk_ZR2*10*y[9]-dk_ZR3*10*y[10]-dk_ZR4*5*y[11]-dk_ZR5*y[12]
      dy[6]=k_ZEB*(y[5]-(5*y[8]+10*y[9]+10*y[10]+5*y[11]+y[12]))-kd_ZEB*y[6]
      dy[7]=k0_200+k_200/(1+((y[2]/J1_200))^3+((y[6]/J2_200))^2)-kd_200*(y[7]-(5*y[8]+2*10*y[9]+3*10*y[10]+4*5*y[11]+5*y[12])-y[15])-dk_ZR1*5*y[8]-dk_ZR2*2*10*y[9]-dk_ZR3*3*10*y[10]-dk_ZR4*4*5*y[11]-dk_ZR5*5*y[12]+lamda1*dk_ZR1*5*y[8]+lamda2*dk_ZR2*2*10*y[9]+lamda3*dk_ZR3*3*10*y[10]+lamda4*dk_ZR4*4*5*y[11]+lamda5*dk_ZR5*5*y[12]-kd_tgfR*y[15]+lamdatgfR*kd_tgfR*y[15]
      dy[8]=Timescale*(K1*(y[7]-(5*y[8]+2*10*y[9]+3*10*y[10]+4*5*y[11]+5*y[12])-y[15])*(y[5]-(5*y[8]+10*y[9]+10*y[10]+5*y[11]+y[12]))-y[8])
      dy[9]=Timescale*(K2*(y[7]-(5*y[8]+2*10*y[9]+3*10*y[10]+4*5*y[11]+5*y[12])-y[15])*y[8]-y[9])
      dy[10]=Timescale*(K3*(y[7]-(5*y[8]+2*10*y[9]+3*10*y[10]+4*5*y[11]+5*y[12])-y[15])*y[9]-y[10])
      dy[11]=Timescale*(K4*(y[7]-(5*y[8]+2*10*y[9]+3*10*y[10]+4*5*y[11]+5*y[12])-y[15])*y[10]-y[11])
      dy[12]=Timescale*(K5*(y[7]-(5*y[8]+2*10*y[9]+3*10*y[10]+4*5*y[11]+5*y[12])-y[15])*y[11]-y[12])
      dy[13]=k_tgf-kd_tgf*(y[13]-y[15])-kd_tgfR*y[15]
      dy[14]=k_OT+k_TGF*(y[13]-y[15])-kd_TGF*y[14]
      dy[15]=Timescale*(TGF_flg+KTGF*(y[7]-(5*y[8]+2*10*y[9]+3*10*y[10]+4*5*y[11]+5*y[12])-y[15])*(y[13]-y[15])-y[15])
      dy[16]=GE*(k_ecad0+k_ecad1/(((y[2]/J_ecad1))^2+1)+k_ecad2/(((y[6]/J_ecad2))^2+1)-kd_ecad*y[16])
      dy[17]=k_ncad0+k_ncad1*(((y[2]/J_ncad1))^2)/(((y[2]/J_ncad1))^2+1)+k_ncad2*(((y[6]/J_ncad2))^2)/(((y[6]/J_ncad2)^2+1)*(1+y[19]/J_ncad3))-kd_ncad*y[17]
      dy[18]=k0O+kO/(1+((y[6]/J_O))^nzo)-kdO*y[18]
      dy[19]=kOp*y[18]-kd_Op*y[19]
    end
    # For Fig 9B
    u0 = [0.128483;1.256853;0.0030203;0.0027977;0.0101511;0.0422942;0.2391346;0.0008014;0.0001464;2.67e-05;4.8e-6;9e-7;0.0619917;1.2444292;0.0486676;199.9383546;137.4267984;1.5180203;1.5180203]
    return ContinuousDynamicalSystem(f, u0, nothing)
end

ds = oval2deterministic()
u0 = get_state(ds)
# Many solutions are unstable :(
u0 = u0 .* (1 .+ 1e-6rand(dimension(ds)))

tr = trajectory(ds, 1000.0, u0; Δt = 0.01, diffeq = (alg = Rodas5(), abstol = 1e-6, reltol = 1e-6))

fig = Figure()
lines(fig[1,1], tr[:, 1])
lines!(fig[1,1], tr[:, 2])
lines(fig[1,2], tr[:, 3], tr[:, 4])
display(fig)