# Hyperparameter Annealing in BEAD

This document explains the new flexible hyperparameter annealing options available in BEAD, with usage instructions, configuration examples, and guidance for adding new strategies.

---

## 1. Overview

BEAD now supports advanced annealing (scheduling) of any hyperparameter, not just learning rate. You can:
- Choose which parameters to anneal (e.g., `lr`, `reg_param`, `contrastive_weight`, etc.)
- Select the annealing strategy per parameter:
  - **constant**: Anneal at a fixed pace every epoch
  - **trigger**: Anneal based on a trigger (e.g., loss plateau, early stopping counter)
  - **hardcoded**: Anneal at specific epochs
- Use trigger-based annealing for any parameter, including non-optimizer ones, using the early stopping counter.

---

## 2. Configuration Examples

Add the following to your config (e.g., in `set_config`):

```python
# === Annealing hyperparameters ===
c.annealing = {
    "lr": {
        "strategy": "trigger",  # options: "constant", "trigger", "hardcoded"
        "pace": 0.95,           # for "constant": multiply by this every epoch
        "trigger": {            # for "trigger": use loss plateau, etc.
            "type": "plateau",
            "patience": 10,
            "factor": 0.5,
            "min_lr": 1e-6,
        },
        "schedule": {           # for "hardcoded": epoch: value
            2: 0.0005,
            3: 0.0001,
        }
    },
    "reg_param": {
        "strategy": "hardcoded",
        "schedule": {
            2: 0.0005,
            3: 0.0001,
        }
    },
    "contrastive_weight": {
        "strategy": "trigger",
        "trigger_values": [0.01, 0.001, 0.0001],  # values to cycle through on each trigger
    }
}
```

---

## 3. Strategy Details

### a. Constant Annealing
- Multiplies the parameter by `pace` every epoch.
- Example:
  ```python
  "lr": {
      "strategy": "constant",
      "pace": 0.95
  }
  ```

### b. Trigger Annealing
- For `lr`, uses PyTorch's ReduceLROnPlateau (loss plateau).
- For other parameters (e.g., `contrastive_weight`), uses the early stopping counter:
  - When the early stopping counter reaches about 2/3 of its patience, the parameter is set to the next value in `trigger_values` (cycling through the list).
  - Works even if patience or total epochs is 1.
- Example:
  ```python
  "contrastive_weight": {
      "strategy": "trigger",
      "trigger_values": [0.01, 0.001, 0.0001]
  }
  ```

### c. Hardcoded Annealing
- Sets the parameter to a specific value at specific epochs.
- Example:
  ```python
  "reg_param": {
      "strategy": "hardcoded",
      "schedule": {
          2: 0.0005,
          3: 0.0001
      }
  }
  ```

---

## 4. Adding a New Annealing Strategy for a Parameter

Suppose you want to add a new annealing strategy for a parameter called `dropout_rate` that linearly decreases from 0.5 to 0.1 over the course of training.

1. **Add to your config:**
   ```python
   "dropout_rate": {
       "strategy": "linear",
       "start": 0.5,
       "end": 0.1
   }
   ```

2. **Implement in code:**
   - In `helper.py` (inside the `Annealer.step` method), add:
     ```python
     elif strat == "linear":
         start = settings.get("start", 0.5)
         end = settings.get("end", 0.1)
         total_epochs = getattr(self.config, "epochs", 1)
         if total_epochs <= 1:
             value = end
         else:
             value = start + (end - start) * (epoch / (total_epochs - 1))
         setattr(self.config, param, value)
     ```
   - This will linearly interpolate the value from `start` to `end` over the epochs.

3. **Example config block:**
   ```python
   "dropout_rate": {
       "strategy": "linear",
       "start": 0.5,
       "end": 0.1
   }
   ```

---

## 5. Full Example Config

```python
c.annealing = {
    "lr": {
        "strategy": "trigger",
        "pace": 0.95,
        "trigger": {"type": "plateau", "patience": 10, "factor": 0.5, "min_lr": 1e-6},
        "schedule": {2: 0.0005, 3: 0.0001}
    },
    "reg_param": {
        "strategy": "hardcoded",
        "schedule": {2: 0.0005, 3: 0.0001}
    },
    "contrastive_weight": {
        "strategy": "trigger",
        "trigger_values": [0.01, 0.001, 0.0001]
    },
    "dropout_rate": {
        "strategy": "linear",
        "start": 0.5,
        "end": 0.1
    }
}
```

---

## 6. Notes
- You can combine multiple strategies for different parameters.
- For trigger-based annealing, the early stopping counter is used for all non-lr parameters.
- The system is robust to short runs (e.g., 1 epoch or 1 patience).
- To add more strategies, simply extend the `Annealer.step` method in `helper.py`.

---

## 7. Troubleshooting
- If a parameter is not changing as expected, check that its strategy and required fields are set in the config.
- For custom strategies, ensure you update both the config and the `Annealer` logic.

---

For further help, see the code in `bead/src/utils/helper.py` and the config examples above.
