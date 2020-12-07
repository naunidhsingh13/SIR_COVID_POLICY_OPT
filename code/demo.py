import math
from Optimizer import StateOptimiser
# from Processor import Processor
# #
obj = StateOptimiser("California")
# #
obj.optimise_state_data()
#
d1, d2, d3 = obj.get_projection_data({})
for x, y, z in zip(d1, d2, d3):
    print(x, " : ", y, " : ", z)
print(obj.theta)

import matplotlib.pyplot as plt

plt.plot(list(range(1, len(d3)+1)), d2)
plt.plot(list(range(1, len(d3)+1)), d3)

plt.show()
obj.dump()
# #
# # #
# p = Processor()
# p.load_all_states()
# # print(p.get_analysis_of("Illinois", {}))
# """mask, social_distance, transit_stations, groc_pharma,
#               retail_recreation, sd_intent, mask_intent, workplace,
#               parks"""
# # t1, t2, t3, t4 = p.get_state_analysis_with_policy_list("Illinois", ["mask"], [])
# # print(t1.values.tolist())
#
# # obj = StateOptimiser("Illinois")
# #
# # obj.optimise_state_data()
# #
# # d1, d2 = obj.get_projection_data({})
# # for x,y in zip(d1, d2):
# #     print(x, " : ", y)
# # print(obj.theta)
# #
# # obj.dump()
#
# # p = Processor()
# # p.load_all_states()
# # "social_distance": 40,
# print(p.get_analysis_of("Illinois", {"workplaces_percent_change_from_baseline": 5}))
# print(p.stateDatas["Illinois"].theta)



def get_gaussian_weights(mu, sigma, rng=(-5, 6)):

    def gauss(x):
        nonlocal mu, sigma
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp((-1 / 2) * ((x - mu) / sigma) ** 2)

    weights = []
    for x in range(rng[0], rng[1]):
        weights.append(gauss(x))

    s = sum(weights)
    weights = [w/s for w in weights]
    return  weights


# print(get_gaussian_weights(0, 1))


