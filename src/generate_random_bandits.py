import re
from argparse import ArgumentParser
from random import randint, random

from options import T_HORIZON
from reward_model import exp, poly

def clip(value,minimum,maximum):
    return max( min( maximum,value ), minimum )


MODEL_FILE = "src/model.py"

# parsing argv
description = "generate random bandits and save them in model.py"
parser = ArgumentParser(description=description)
parser.add_argument("-N", help='how many bandits to create', metavar=1, required=True)

args, _ = parser.parse_known_args()

# read and keep the old content of model.py
with open(MODEL_FILE, "r") as f:
    buffer = f.read()

last_idx = int( re.findall( "\d+", re.findall("\d+\s?:\s?\[", buffer)[-1] )[0] )
buffer = buffer[:-2] # remove "}\n"

# create random bandits
for i in range(int(args.N)):
    buffer += f"\t{i+last_idx+1} : [\n"
    for _ in range(randint(2,15)):
        fun, descr = exp(randint(2,5), randint(2,T_HORIZON[0]), clip(random(),1e-2,1-1e-2)) if randint(0,1) else poly(clip(random(),4e-1,1), randint(2,T_HORIZON[0]), clip(random(),1e-2,1-1e-2))
        buffer += f"\t\tlambda x: {descr},\n"
    buffer += "\t],\n\n"

buffer += "}\n"

with open(MODEL_FILE, "w") as f:
    f.write(buffer)

print("You can now find your randomly generated bandits in", MODEL_FILE)