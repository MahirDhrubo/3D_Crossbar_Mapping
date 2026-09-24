* Numerically robust VTEAM voltage-threshold memristor subcircuit for Xyce
*
* Source model:
*   Kvatinsky et al., "VTEAM: A General Model for Voltage-Controlled
*   Memristors," IEEE TCAS-II, 2015.
*
* Terminals:
*   p, n : memristor electrical terminals
*   x    : normalized internal-state monitor
*          V(x)=0 -> w=won/R=Ron
*          V(x)=1 -> w=woff/R=Roff
*
* Sign convention follows the paper:
*   v(p,n) > voff drives w toward woff and raises resistance.
*   v(p,n) < von  drives w toward won  and lowers resistance.
*
* Implements the VTEAM state law with a linear R(w) relation (paper Eq. 7)
* and directional rectangular boundary checks.
*
* Defaults are from Table II, "VTEAM fit to the Yakopcic model."

.SUBCKT VTEAM p n x PARAMS:
+ ron=387
+ roff=1069.5
+ won=0
+ woff=10n
+ winit=8.9n
+ voff=0.16
+ von=-0.15
+ koff=2.49e-6
+ kon=-2.2e-4
+ alphaoff=4
+ alphaon=4

* Normalized state s = (w-won)/(woff-won).
* logic = 1, s = 0 -> w = won -> Ron -> LRS
* logic = 0, s = 1 -> w = woff -> Roff -> HRS
* Linear VTEAM resistance: R = RON + (ROFF-RON)*s.
Bmem p n I={V(p,n)/(ron + (roff-ron)*MIN(MAX(V(x),0),1))}

* One-farad numerical integrator: V(x) is the normalized state s.
Cstate x 0 1 IC={MIN(MAX((winit-won)/(woff-won),0),1)}
Rstate_leak x 0 1e15

* ds/dt = (dw/dt)/(woff-won).
* IF() keeps relational checks in Xyce's supported expression form.
* MAX(...,0) prevents invalid fractional-power arguments.
Bstate x 0 I={-IF(V(p,n)>voff,
+ IF(V(x)<1,
+ (koff/(woff-won))*POW(MAX(V(p,n)/voff-1,0),alphaoff),
+ 0),
+ IF(V(p,n)<von,
+ IF(V(x)>0,
+ (kon/(woff-won))*POW(MAX(V(p,n)/von-1,0),alphaon),
+ 0),
+ 0))}

.ENDS VTEAM
