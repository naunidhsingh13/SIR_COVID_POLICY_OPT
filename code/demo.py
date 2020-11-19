
from Optimizer import StateOptimiser
from Processor import Processor
#
<<<<<<< HEAD
# obj = StateOptimiser("Illinois")
#
# obj.optimise_state_data()
#
# d1, d2 = obj.get_projection_data({})
# for x,y in zip(d1, d2):
#     print(x, " : ", y)
# print(obj.theta)
#
# obj.dump()
#
# #
p = Processor()
p.load_all_states()
# print(p.get_analysis_of("Illinois", {}))
t1, t2, t3, t4 = p.get_state_analysis_with_policy_list("Illinois", [], [])
print(t1.values.tolist())

=======
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
>>>>>>> da202083c4b802985f2853cc96b1aeeaa0d44138
