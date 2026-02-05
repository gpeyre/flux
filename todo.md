This is a project in pytorch (so GPU-based to accelerate simulation computation) to make an interative dynamic particule simulation with a gaming aims (the user will interact, and ultimately there will be goals and reward, but first we focuss on basics ).

Maybe you should use pygame or similar framework to handle rendering and use interaction with the playground (displacing stuff on the playground) and with the sliders

The playground for the simulation is a square with reflecting boundary condition (particules bounce on it). 

The particular moves according to an ODE inegrator real time. It moves accordint to newtonian law with a friction coefficient with a slider on the right to tune it (if no friction : just equation ddot x_i = -f(x_1...x_n) where the xi are 2D particules loction and f is all the force, external and internals)

There is a slider to accelerate/slowdown the simulation time (integration step size).

There is no gravity, and particules moves according to external forces which are radial (which the user can modify by displacing the centers) and self interaction (which are both attractive long range and repultives short range) which the user can modify by sliders.

The center of external force are indicarted by a circle depicting the average range of action (they have a decaying gaussian-type decay of influence, and the radius is proportional to the bandwith) and blue/red if they are attractive/repulsive.

The user can modify the number of particular between 1 particule and 300 by a sliders. When moving the sliders, particules are killed at random to lower the number and created at random position when increasing the number.

The sliders are located on the right of the screen, next to the playground. 

Be sure to make a very clean repository, putting auxiliary needed functions in a toolbox, a nice readme presenting project and usage, etc. It should be very professional. 