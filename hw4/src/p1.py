import numpy as np

# prev_layer, left num units, total layer

num_unit = 36
MAX_LAYER = 19
dp = np.ones((num_unit, num_unit, MAX_LAYER)) * -1

for prev_layer in range(1, num_unit):

    dp[prev_layer, num_unit - prev_layer - 1, 1] = 10 * prev_layer

dp[num_unit - 1, 0, 1] = 10 * (num_unit - 1) + num_unit

for l in range(2, MAX_LAYER):
    for prev_layer in range(1, num_unit): # prev_layer
        for left_units in range(1, num_unit): # left units

            if prev_layer + left_units + 1 > num_unit:
                break

            for cur_layer in range(1, left_units):

                if dp[prev_layer, left_units, l - 1] < 0:
                    break

                if cur_layer + 1 == left_units:     # last layer
                    gain = (prev_layer + 1) * cur_layer + (cur_layer + 1)
                else:                               # middle layer
                    gain = (prev_layer + 1) * cur_layer

                dp[prev_layer, left_units - cur_layer - 1, l] = \
                        max(
                            dp[prev_layer, left_units - cur_layer - 1, l],
                            dp[prev_layer, left_units, l - 1] + gain
                        )
# 36, 36, 19
for i in range(36):
    for j in range(36):
        for k in range(19):
            if dp[i, j, k] == 510:
                print(i, j, k)
print(np.max(np.max(dp, axis=0), axis=0))
