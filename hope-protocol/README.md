# Hope Protocol

Stochastic reward amplification to prevent premature convergence in learning systems.

# Agents: How the AI Learns

q_learning.py contains the Q-learning agent. This is the part responsible for things like the Q-table, choosing actions, updating Q-values, epsilon exploration, and learning from rewards.

# envs/: The "world" the Agent Experiences

hope_reward_wrapper.py is especially important. FrozenLake normally provides its own rewards, but the wrapper gives you a place to intercept or modify that reward behavior before the agent receives it.

Conceptually:

    FrozenLake
        ↓
    HOPE reward wrapper
        ↓
    Q-learning agent

At this point, there is NO rewriting of ForzenLakes itself.

# hope/: Experimental Idea

hope_protocol.py

This should contain the actual HOPE mechanism aka what's being tested to determine whether selectively emphasizing rare positive outcomes changes learning behavior.

This separation is important academically, too:

- Q-learning = established algorithm
- FrozenLake = established environment
- HOPE Protocol = your experimental intervention

That makes it much easier to ask:

- Does standard Q-learning behave differently when HOPE is introduced?

# experiments/: The Test

This directory is essentially the laboratory.

For example:

| run_frozenlake_baseline.py

should mean:

| Run ordinary Q-learning without HOPE.

While:

| run_frozenlake_hope.py

means:

| Run the comparable experiment with HOPE enabled.

And then:

| compare_stability.py

lets you compare the resulting behavior.

# requirements.txt

Should be ran before testing to set install necessary packages and ensure accurate testing.

---

---

# start-up

## Running HOPE Experiments

Open PowerShell or the VS Code terminal and navigate to the root of the HOPE Protocol project:

```powershell
cd C:\Users\Aerusuke\Desktop\Hope-Protocol\hope-protocol
```

### 1. Install the required Python packages

If this is your first time running the project:

```powershell
py -3.11 -m pip install numpy gymnasium
```

### 2. Run an experiment

Experiments should be run **from the project root as Python modules**.

For example:

```powershell
py -3.11 -m experiments.compare_stability
```

Do **not** run:

```powershell
python experiments/compare_stability.py
```

Running the file directly may cause errors such as:

```text
ModuleNotFoundError: No module named 'agents'
```

because the experiment imports modules from other project folders.

### Other experiments

Use the same format for other files inside `experiments/`:

```powershell
py -3.11 -m experiments.FILE_NAME
```

For example, if the file is `compare_compatibility.py`:

```powershell
py -3.11 -m experiments.compare_compatibility
```

Do not include `.py` when using `python -m`.
