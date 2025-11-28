from agents.fault_agent import run_conversation

prompt = """
Calculate v at each bus
yBus = [3-8.95*1j -2+6*1j -1+3*1j 0
-2+6*1j 3.774-11.306*1j -0.674+2.024*1j -1.044+3.134*1j
-1+3*1j -0.674+2.024*1j 3.666-10.96*1j -2+6*1j
0 -1.044+3.134*1j -2+6*1j 3-8.99*1j ];

V_pre = Bus 1 : 1.0500 +0.0000j   (slack bus)
Bus 2 : 0.9857 -0.0001j
Bus 3 : 0.9936 +0.1129j
Bus 4 : 1.0123 +0.0386j

Fault bus at 4

"""

print(run_conversation(prompt))