import argparse
import random
import subprocess
import shlex
import sys
import os
import math

parser = argparse.ArgumentParser(description='optimizing model parameters')
parser.add_argument('--type', choices=['random', 'genetic', 'anneal'],
                    default='random', help='gives the type of optimizer, can be random, genetic, or anneal (simulated annealing)')
args = parser.parse_args()

fileName = "pre-trained/formation-energy-per-atom.pth.tar"

epochs = [50, 100, 150, 200, 250]
batchsize = [64, 128, 256, 384, 512]
learningrate = [0.1, 0.01, 0.001, 0.0001]
lrmilestone = [25, 50, 100]
momentum = [0.8, 0.9, .925, .95, .975, 0.99]
weightdecay = [0, 0.1, 0.01, 0.0001, 0.00001]
#trainsize = [100, 200, 300, 400, 500, 600]
atomfealen = [8, 16, 32, 64, 128]
hfealen = [8, 16, 32, 64, 128, 256]
nconv = [2, 3, 4, 5]
nh = [1, 2, 3]
freezeconv = [True, False]
freezeemb = [True, False]
freezefc = [0, 1, 2]

low = sys.float_info.max
result_dir = os.path.expanduser('~/dgnn/cgcnndefect/model.all')
data_dir = os.path.expanduser('~/dgnn/cgcnndefect/examples/OxideMLpaper1/cgcnn')

# Ensure the result and data directories exist
os.makedirs(result_dir, exist_ok=True)
assert os.path.exists(data_dir), f'data_dir does not exist: {data_dir}'

if args.type == 'random':
    for x in range(100):

        #r1 = int(random.random() * len(epochs))
        r2 = int(random.random() * len(batchsize))
        r3 = int(random.random() * len(learningrate))
        r4 = int(random.random() * len(lrmilestone))
        r5 = int(random.random() * len(momentum))
        r6 = int(random.random() * len(weightdecay))
        #r7 = int(random.random() * len(trainsize))
        r9 = int(random.random() * len(atomfealen))
        r10 = int(random.random() * len(hfealen))
        r11 = int(random.random() * len(nconv))
        r12 = int(random.random() * len(nh))
        r14 = int(random.random() * len(freezeconv))
        r15 = int(random.random() * len(freezeemb))
        r16 = int(random.random() * len(freezefc))

        stringstring = (
            f"python -m cgcnndefect.command_line_train --epochs 100 "
            f"--batch-size {batchsize[r2]} "
            f"--lr {learningrate[r3]} "
            f"--lr-milestones {' '.join(map(str, [lrmilestone[r4]]))} "
            f"--momentum {momentum[r5]} "
            f"--weight-decay {weightdecay[r6]} "
            f"--atom-fea-len {atomfealen[r9]} "
            f"--h-fea-len {hfealen[r10]} "
            f"--n-conv {nconv[r11]} "
            f"--n-h {nh[r12]} "
            f"{'--freeze-conv ' if freezeconv[r14] else ''}"
            f"{'--freeze-embedding ' if freezeemb[r15] else ''}"
            f"--freeze-fc {freezefc[r16]} "
            f"--atom-spec locals_continuous --crys-spec globals "
            f"--resultdir {result_dir} --csv-ext .all --init-embed-file atom_init.json "
            f"--disable-cuda {data_dir}"
        )

        f = subprocess.run(shlex.split(stringstring), encoding='utf-8', stdout=subprocess.PIPE)
        losses = []
        for line in f.stdout.split('\n'):
            if 'Loss' in line:
                afterloss = line.split('Loss')[1]
                loss = float(afterloss.split()[0].strip())
                losses.append(loss)

        if losses:
            print(losses)
            print(losses[-1])  # Use -1 to get the last element
            if losses[-1] < low:
                with open("optimized_python_command.txt", "w") as g:
                    g.write(stringstring + "\n" + str(losses[-1]))
                low = losses[-1]
        else:
            print("No losses found in the output.")
        print('ITERATION ' + str(x) + ' DONE')

def momentumchoice(currmomentum):
    if currmomentum == 0.5:
        currmomentum = currmomentum + random.random() / 4
    else:
        x = random.random()
        if x < 0.5:
            distancetohalf = currmomentum - 0.5
            currmomentum = currmomentum - 0.5 * random.random() * distancetohalf
        else:
            distanceto1 = 1 - currmomentum
            currmomentum = currmomentum + 0.25 * random.random() * distanceto1
    return currmomentum

def weightdecaychoice(currwd):
    if currwd == 0:
        x = random.random()
        if x < 0.5:
            currwd = 0.01
        else:
            currwd = 0.1
    else:
        x = random.random()
        if x < 0.3:
            currwd = currwd / 10
        elif x < 0.7:
            currwd = currwd
        else:
            multto1 = 1 / currwd
            currwd = currwd * random.random() * 0.5 * multto1
    return currwd

def trainsizechoice(curr):
    x = random.random()
    if curr == 700:
        if x < 0.5:
            return curr
        else:
            return 600
    elif curr == 100:
        if x < 0.5:
            return curr
        else:
            return 200
    else:
        if x < 0.3:
            return curr - 100
        elif x < 0.7:
            return curr
        else:
            return curr + 100

def fealenchoice(curr):
    x = random.random()
    if curr == 8:
        if x < 0.5:
            return curr
        else:
            return 16
    else:
        if x < 0.3:
            return curr // 2  # Use integer division
        elif x < 0.7:
            return curr
        else:
            return curr * 2

def convchoice(curr):
    x = random.random()
    if curr == 3:
        if x < 0.5:
            return curr
        else:
            return curr + 1
    else:
        distto3 = curr - 3
        if x < 0.3:
            return curr - int(random.random() * (distto3 + 1))
        elif x < 0.7:
            return curr
        else:
            return curr + 1 + int(random.random() * 2)

def hiddenchoice(curr):
    x = random.random()
    if curr == 1:
        if x < 0.5:
            return curr
        else:
            return curr + 1
    else:
        distto1 = curr - 1
        if x < 0.3:
            return curr - int(random.random() * (distto1 + 1))
        elif x < 0.7:
            return curr
        else:
            return curr + 1 + int(random.random() * 2)

result_dir = os.path.expanduser('~/dgnn/cgcnndefect/model.all')
data_dir = os.path.expanduser('~/dgnn/cgcnndefect/examples/OxideMLpaper1/cgcnn')

# Ensure result directory exists
os.makedirs(result_dir, exist_ok=True)

# Placeholder argparse mock for args
class Args:
    type = 'anneal'
args = Args()

if args.type == 'anneal':
    temp = 1
    count = 0

    curr = [10, 256, 0.01, 50, 0.9, 0, 600, 8, 16, 4, 2]
    curr = [int(i) for i in curr]

    # Create the command string with additional parameters
    stringstring = (
        f"python -m cgcnndefect.command_line_train --epochs {curr[0]} --batch-size {curr[1]} "
        f"--learning-rate {curr[2]} --weight-decay {curr[5]} --optim Adam --atom-fea-len {curr[7]} "
        f"--h-fea-len {curr[8]} --n-conv {curr[9]} --atom-spec locals_continuous --crys-spec globals "
        f"--resultdir {result_dir} --csv-ext .all --init-embed-file atom_init.json --disable-cuda {data_dir}"
    )

    print(f"Initial command: {stringstring}")
    # Run the command and capture the output
    f = subprocess.run(shlex.split(stringstring), encoding='utf-8', stdout=subprocess.PIPE)
    losses = []

    # Parse the output for loss values
    for line in f.stdout.split('\n'):
        if 'Loss' in line:
            afterloss = line.split('Loss')[1]
            loss = float(afterloss.split()[0].strip())
            losses.append(loss)

    if losses:
        currloss = losses[-1]
    else:
        print("No loss values found in the output.")
        exit(1)

    while temp > 0.0001:
        new = [
            curr[0] + int((random.random() - 0.5) * 10),
            curr[1] + int((random.random() - 0.5) * 48),
            curr[2] / int(random.random() * 100 + 1),
            weightdecaychoice(curr[5]),
            trainsizechoice(curr[6]),
            fealenchoice(curr[7]),
            fealenchoice(curr[8]),
            convchoice(curr[9])
        ]
        
        new = [int(n) if isinstance(n, float) else n for n in new]  # Ensure all values are integers

        # Create the new command string with additional parameters
        stringstring = (
            f"python -m cgcnndefect.command_line_train --epochs 10 --batch-size {new[1]} "
            f"--learning-rate {new[2]} --weight-decay {new[3]} --optim Adam "
            f"--atom-fea-len {new[5]} --h-fea-len {new[6]} --n-conv {new[7]} --atom-spec locals_continuous "
            f"--crys-spec globals --resultdir {result_dir} --csv-ext .all --init-embed-file atom_init.json "
            f"--disable-cuda {data_dir}"
        )

        print(f"Running command: {stringstring}")
        # Run the new command and capture the output
        f = subprocess.run(shlex.split(stringstring), encoding='utf-8', stdout=subprocess.PIPE)
        losses = []

        # Parse the output for loss values
        for line in f.stdout.split('\n'):
            if 'Loss' in line:
                afterloss = line.split('Loss')[1]
                loss = float(afterloss.split()[0].strip())
                losses.append(loss)

        if losses:
            newloss = losses[-1]
            print(f"Losses: {losses}")
            print(f"New loss: {newloss}")
        else:
            print("No loss values found in the output for new configuration.")
            continue

        accept = math.exp((currloss - newloss) / temp)
        print(f"Acceptance probability: {accept}")

        if accept > 1 or random.random() < accept:
            with open("optimized_python_command.txt", "w") as g:
                g.write(stringstring + "\n" + str(newloss))
            currloss = newloss
            print(f"Accepted new configuration with loss: {newloss}")

        temp = 1 / (count + 1)
        count += 1
        print(f"Iteration {count} done, temp: {temp}")

print('DONE')
