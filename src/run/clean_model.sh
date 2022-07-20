text="from math import e

lambdas = {
    0: [   # this is the 15 arms bandit used in our experiments
        lambda x: 1.0*(1 - 177 / (x+177**(1/0.8))**0.8),
        lambda x: 0.91*(1 - 571 / (x+571**(1/0.9))**0.9),
        lambda x: 0.73*(1 - e**(-3*x/162)),
        lambda x: 0.16*(1 - e**(-3*x/949)),
        lambda x: 0.48*(1 - 144 / (x+144**(1/1.0))**1.0),
        lambda x: 0.43*(1 - 195 / (x+199**(1/0.9))**0.9),
        lambda x: 0.79*(1 - 695 / (x+695**(1/0.9))**0.9),
        lambda x: 0.62*(1 - e**(-1*x/271)),
        lambda x: 0.23*(1 - e**(-1*x/849)),
        lambda x: 0.39*(1 - e**(-5*x/457)),
        lambda x: 0.37*(1 - 290 / (x+290**(1/1.0))**1.0),
        lambda x: 0.17*(1 - 223 / (x+223**(1/1.0))**1.0),
        lambda x: 1.08*(1 - 85 / (x+85**(1/0.8))**0.8),
        lambda x: 0.84*(1 - 219 / (x+219**(1/1.0))**1.0),
        lambda x: 0.26*(1 - 124 / (x+124**(1/0.9))**0.9),
    ],

    1: [   # this is the 2 arms bandit used in our experiments
        lambda x: 0.4*(1 - (50000 / (100*x + 50000))),
        lambda x: x / 25000 if x < 25000 else 1
    ],
}"

echo "$text" > src/model.py
