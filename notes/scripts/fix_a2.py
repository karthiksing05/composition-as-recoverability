import sys
p = sys.argv[1]; s = open(p, encoding="utf-8").read()
R = [
(r"and its exact value is given by the covariance form of Proposition~\ref{prop:ceiling}.",
 r"and the covariance form of Proposition~\ref{prop:ceiling} gives 0.66, averaged over random projections."),
(r"By Definition~\ref{def:recoverable}, then, the states of these networks are recoverable whatever their vocabulary,",
 r"Their role-factorization residual, measured in Section~\ref{sec:structure}, is $\delta = 0.02$--$0.04$, so by Definition~\ref{def:repcomp} these states are compositional at a tolerance of about $\tau = 0.27$ at $V = 8$ and $0.17$ at $V = 64$, and not at $\tau = 0.1$. By Definition~\ref{def:recoverable}, then, the states of these networks are recoverable to this degree whatever their vocabulary,"),
(r"Note that Definition~\ref{def:repcomp} is graded and relative.",
 r"Note that Definition~\ref{def:repcomp} is graded and relative. The normalization is well posed: for a symmetric blend of two parts with identical, independent distributions, no readout can tell the roles apart, and the best affine readout returns the mean of the two parts, so $\varepsilon_{\mathrm{blend}} = 1/2$ in units of part variance, whatever the code."),
]
for a, b in R:
    assert s.count(a) == 1, a[:80]; s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s); print("ok")
