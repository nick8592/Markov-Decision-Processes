Right_Probability = [
    [  0, 0.3, 0.7,   0,   0,   0,   0,   0],
    [0.1,   0,   0, 0.2, 0.7,   0,   0,   0],
    [0.1,   0,   0,   0, 0.2, 0.7,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  0, 0.1, 0.1,   0,   0,   0, 0.2, 0.6],
    [  0,   0, 0.5,   0,   0,   0,   0, 0.5],
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0]
]

Left_Probability = [
    [  0, 0.8, 0.2,   0,   0,   0,   0,   0],
    [0.1,   0,   0, 0.7, 0.2,   0,   0,   0],
    [0.1,   0,   0,   0, 0.7, 0.2,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  0, 0.2, 0.1,   0,   0,   0, 0.5, 0.2],
    [  0,   0, 0.2,   0,   0,   0,   0, 0.8],
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0]
]

Back_Probability = [
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [0.4,   0,   0, 0.2, 0.4,   0,   0,   0],
    [0.4,   0,   0,   0, 0.2, 0.4,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  0, 0.3, 0.3,   0,   0,   0, 0.2, 0.2],
    [  0,   0, 0.9,   0,   0,   0,   0, 0.1],
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0]
]

TransitionProbaility = []
TransitionProbaility.append(Right_Probability)
TransitionProbaility.append(Left_Probability)
TransitionProbaility.append(Back_Probability)