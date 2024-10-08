# Capture the Flag (CtF)

The CtF game is a simple grid-world environment with discrete state and action spaces, but includes complex adversarial dynamics, as shown in \Cref{fig:game_field} inspired by \cite{yuasa2024generating}.

\begin{figure}[tb]
    \centering
    \medskip
    \includegraphics[width=0.5\linewidth]{figs/tl_multigrid.png}
    \caption{Visualization of our 2 vs. 2 CtF game. Black squares are obstacles, triangles are agents, and circles are flags. The region highlighted by solid red lines is the border region for a red agent whose policy is \textit{patrol} policy.}
    \label{fig:game_field}
\end{figure}

In our CtF game, if a pair of blue and red agents are next to each other in the blue territory, then the red agent is killed with 75\% probability (and vice versa in the red territory).
The game ends when either agent captures its opponent's flag or all blue agents are defeated. 
The 12$\times$12 state space is fully observable, and there are 5 discrete actions for an agent: stay, up, right, down, and left.
The field objects consist of the $m$ friendly agents, $n$ enemy agents, 2 flags, and 96 territories (48 for each agent), and 48 obstacles (4 at the center and 44 surrounding the the territories).
The observation of a state is a 12 $\times$ 12 $\times$ 3 tensor, where the first layer represents a map of the territories and obstacles, the second layer represent a map of the agents and flags, and the third layer represents status of the agents whether an agent is dead or alive.
This observation tensor is used as the input to the RL algorithms.

At every timestep, the blue agent is rewarded -0.01 to encourage it to reach the red flag faster.
The blue agent is rewarded 1 by capturing the red flag and -1 by having the blue flag captured by the red agent.
Furthermore, the blue agent is rewarded 0.25 by killing the red agent and -0.25 by being killed by the red agent.