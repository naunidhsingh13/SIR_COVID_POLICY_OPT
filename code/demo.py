
from Optimizer import StateOptimiser
from Processor import Processor
#
obj = StateOptimiser("Illinois")

obj.optimise_state_data()

d1, d2 = obj.get_projection_data({})
for x,y in zip(d1, d2):
    print(x, " : ", y)
print(obj.theta)

obj.dump()

p = Processor()
p.load_all_states()
print(p.get_analysis_of("Illinois", {}))