import matlab.engine

eng = matlab.engine.start_matlab()

# MATLAB symbolic differential equation code
matlab_code = r"""
syms y(t)
Dy = diff(y, t);
ode = Dy == -2*y;

ySol(t) = dsolve(ode, y(0) == 1);
assignin('base', 'y_final', ySol);
"""

eng.eval(matlab_code, nargout=0)

# Fetch symbolic result as string
y_expr = eng.eval("char(y_final)", nargout=1)

print("y(t) =", y_expr)
