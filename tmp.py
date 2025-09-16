# @title Advanced settings

import yaml

config = {}

# @markdown <br/>

# @markdown ## Global
project = "gfp"  # @param {type:"string"}
debug = True  # @param {type:"boolean"}
device = "cuda"  # @param ["cuda","cpu"]
seed = 42  # @param {type:"integer"}

config = {
    "project": project,
    "debug": debug,
    "device": device,
    "seed": seed,
}

# @markdown <br/>

# @markdown ## Sampler
num_shuffles = 100000  # @param {type:"integer"}
shuffle_rate = 0.04  # @param {type:"number"}
window_sizes = [1, 3, 5]  # @param {type:"raw"}

config["sampler"] = {
    "num_shuffles": num_shuffles,
    "shuffle_rate": shuffle_rate,
    "window_sizes": window_sizes,
}

config["predictor"] = {}

# @markdown <br/>

# @markdown ## Predictor (Value1)
name = "ex"  # @param {type:"string"}
batch_size = 16  # @param {type:"integer"}
destruct_per_samples = 0  # @param {type:"integer"}
model_name_or_path = "facebook/esm2_t30_150M_UR50D"  # @param {type:"string"}
mutate_per_samples = 150  # @param {type:"integer"}
noise_ratio = 0.1  # @param {type:"number"}
num_destructions = 2  # @param {type:"integer"}
num_epochs = 200  # @param {type:"integer"}
num_mutations = 2  # @param {type:"integer"}
num_trials = 30  # @param {type:"integer"}
patience = 20  # @param {type:"integer"}
test_size = 0.2  # @param {type:"number"}

if name:
    config["predictor"][name] = {
        "batch_size": batch_size,
        "destruct_per_samples": destruct_per_samples,
        "model_name_or_path": model_name_or_path,
        "mutate_per_samples": mutate_per_samples,
        "noise_ratio": noise_ratio,
        "num_destructions": num_destructions,
        "num_epochs": num_epochs,
        "num_mutations": num_mutations,
        "num_trials": num_trials,
        "patience": patience,
        "test_size": test_size,
    }

# @markdown <br/>

# @markdown ## Predictor (Value2)
name = "em"  # @param {type:"string"}
batch_size = 16  # @param {type:"integer"}
destruct_per_samples = 0  # @param {type:"integer"}
model_name_or_path = "facebook/esm2_t30_150M_UR50D"  # @param {type:"string"}
mutate_per_samples = 150  # @param {type:"integer"}
noise_ratio = 0.1  # @param {type:"number"}
num_destructions = 2  # @param {type:"integer"}
num_epochs = 200  # @param {type:"integer"}
num_mutations = 2  # @param {type:"integer"}
num_trials = 30  # @param {type:"integer"}
patience = 20  # @param {type:"integer"}
test_size = 0.2  # @param {type:"number"}

if name:
    config["predictor"][name] = {
        "batch_size": batch_size,
        "destruct_per_samples": destruct_per_samples,
        "model_name_or_path": model_name_or_path,
        "mutate_per_samples": mutate_per_samples,
        "noise_ratio": noise_ratio,
        "num_destructions": num_destructions,
        "num_epochs": num_epochs,
        "num_mutations": num_mutations,
        "num_trials": num_trials,
        "patience": patience,
        "test_size": test_size,
    }

# @markdown <br/>

# @markdown ## Predictor (Value3)
name = "brightness"  # @param {type:"string"}
batch_size = 16  # @param {type:"integer"}
destruct_per_samples = 0  # @param {type:"integer"}
model_name_or_path = "facebook/esm2_t30_150M_UR50D"  # @param {type:"string"}
mutate_per_samples = 150  # @param {type:"integer"}
noise_ratio = 0.1  # @param {type:"number"}
num_destructions = 2  # @param {type:"integer"}
num_epochs = 200  # @param {type:"integer"}
num_mutations = 2  # @param {type:"integer"}
num_trials = 30  # @param {type:"integer"}
patience = 20  # @param {type:"integer"}
test_size = 0.2  # @param {type:"number"}

if name:
    config["predictor"][name] = {
        "batch_size": batch_size,
        "destruct_per_samples": destruct_per_samples,
        "model_name_or_path": model_name_or_path,
        "mutate_per_samples": mutate_per_samples,
        "noise_ratio": noise_ratio,
        "num_destructions": num_destructions,
        "num_epochs": num_epochs,
        "num_mutations": num_mutations,
        "num_trials": num_trials,
        "patience": patience,
        "test_size": test_size,
    }

config["evaluator"] = {}

# @markdown <br/>

# @markdown ## Evaluator (Hamiltonian)
threshold = -5.0  # @param {type:"raw"}
top_p = None  # @param {type:"raw"}
top_k = None  # @param {type:"raw"}
mode = "max"  # @param ["max", "min", "range"]
upper = None  # @param {type:"raw"}
lower = None  # @param {type:"raw"}
config["evaluator"]["hamiltonian"] = {
    "threshold": threshold,
    "top_p": top_p,
    "top_k": top_k,
    "mode": mode,
    "upper": upper,
    "lower": lower,
}

# @markdown <br/>

# @markdown ## Evaluator (Likelihoood)
batch_size = 32  # @param {type:"integer"}
model_name_or_path = "westlake-repl/SaProt_650M_AF2"  # @param {type:"string"}
threshold = 0.0  # @param {type:"raw"}
top_p = None  # @param {type:"raw"}
top_k = None  # @param {type:"raw"}
mode = "max"  # @param ["max", "min", "range"]
upper = None  # @param {type:"raw"}
lower = None  # @param {type:"raw"}

config["evaluator"]["likelihood"] = {
    "batch_size": batch_size,
    "model_name_or_path": model_name_or_path,
    "threshold": threshold,
    "top_p": top_p,
    "top_k": top_k,
    "mode": mode,
    "upper": upper,
    "lower": lower,
}

# @markdown <br/>

# @markdown ## Evaluator (Value1)
name = "ex"  # @param {type:"string"}
batch_size = 32  # @param {type:"integer"}
mode = "range"  # @param ["max", "min", "range"]
upper = 388  # @param {type:"raw"}
lower = 378  # @param {type:"raw"}

# @markdown **Series**
threshold = None  # @param {type:"raw"}
top_p = 0.4  # @param {type:"raw"}
top_k = None  # @param {type:"raw"}

series = {
    "threshold": threshold,
    "top_p": top_p,
    "top_k": top_k,
}

# @markdown **Parallel**
threshold = None  # @param {type:"raw"}
top_p = 0.2  # @param {type:"raw"}
top_k = None  # @param {type:"raw"}

parallel = {
    "threshold": threshold,
    "top_p": top_p,
    "top_k": top_k,
}

if name:
    config["evaluator"][name] = {
        "batch_size": batch_size,
        "mode": mode,
        "upper": upper,
        "lower": lower,
        "series": series,
        "parallel": parallel,
    }

# @markdown <br/>

# @markdown ## Evaluator (Value2)
name = "em"  # @param {type:"string"}
batch_size = 32  # @param {type:"integer"}
mode = "range"  # @param ["range","max","min"]
upper = 453  # @param {type:"integer"}
lower = 443  # @param {type:"integer"}

# @markdown **Series**
threshold = None  # @param {type:"raw"}
top_p = 0.4  # @param {type:"raw"}
top_k = None  # @param {type:"raw"}

series = {
    "threshold": threshold,
    "top_p": top_p,
    "top_k": top_k,
}

# @markdown **Parallel**
threshold = None  # @param {type:"raw"}
top_p = 0.2  # @param {type:"raw"}
top_k = None  # @param {type:"raw"}

parallel = {
    "threshold": threshold,
    "top_p": top_p,
    "top_k": top_k,
}

if name:
    config["evaluator"][name] = {
        "batch_size": batch_size,
        "mode": mode,
        "upper": upper,
        "lower": lower,
        "series": series,
        "parallel": parallel,
    }

# @markdown <br/>

# @markdown ## Evaluator (Value3)
name = "brightness"  # @param {type:"string"}
batch_size = 32  # @param {type:"integer"}
mode = "max"  # @param ["range","max","min"]

# @markdown **Series**
threshold = None  # @param {type:"raw"}
top_p = 0.4  # @param {type:"raw"}
top_k = None  # @param {type:"raw"}

series = {
    "threshold": threshold,
    "top_p": top_p,
    "top_k": top_k,
}

# @markdown **Parallel**
threshold = None  # @param {type:"raw"}
top_p = 0.2  # @param {type:"raw"}
top_k = None  # @param {type:"raw"}

parallel = {
    "threshold": threshold,
    "top_p": top_p,
    "top_k": top_k,
}

if name:
    config["evaluator"][name] = {
        "batch_size": batch_size,
        "mode": mode,
        "series": series,
        "parallel": parallel,
    }

# @markdown <br/>

# @markdown ## Generator
batch_size = 32  # @param {type:"integer"}
max_new_token = 233  # @param {type:"integer"}
model_name_or_path = "hugohrban/progen2-small"  # @param {type:"string"}
num_epochs = 6  # @param {type:"integer"}
num_trials = 30  # @param {type:"integer"}
patience = 3  # @param {type:"integer"}
prompt = "MSKGE"  # @param {type:"string"}
test_size = 0.1  # @param {type:"number"}

config["generator"] = {
    "batch_size": batch_size,
    "max_new_token": max_new_token,
    "model_name_or_path": model_name_or_path,
    "num_epochs": num_epochs,
    "num_trials": num_trials,
    "patience": patience,
    "prompt": prompt,
    "test_size": test_size,
}

# @markdown <br/>

# @markdown ## Early Stopper
batch_size = 32  # @param {type:"integer"}
model_name_or_path = "facebook/esm2_t33_650M_UR50D"  # @param {type:"string"}
num_samples = 1000  # @param {type:"integer"}
patience = 5  # @param {type:"integer"}

config["early_stopper"] = {
    "batch_size": batch_size,
    "model_name_or_path": model_name_or_path,
    "num_samples": num_samples,
    "patience": patience,
}

# @markdown <br/>

# @markdown ## Runner
num_iterations = 35  # @param {type:"integer"}
num_sequences = 20000  # @param {type:"integer"}

config["runner"] = {
    "num_iterations": num_iterations,
    "num_sequences": num_sequences,
}

with open("config.yaml", "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
