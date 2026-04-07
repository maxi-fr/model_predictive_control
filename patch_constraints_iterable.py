with open("src/model_predictive_control/constraints.py") as f:
    content = f.read()

# Fix string iterable handling in resolve_indices
old_iter = """        if isinstance(time_indices, Iterable):
            return [i if i >= 0 else N + 1 + i for i in time_indices]
        raise ValueError(f"Unsupported time_indices format: {type(time_indices)}")"""

new_iter = """        if isinstance(time_indices, Iterable) and not isinstance(time_indices, str):
            return [i if i >= 0 else N + 1 + i for i in time_indices]
        raise ValueError(f"Unsupported time_indices format: {type(time_indices)}")"""
content = content.replace(old_iter, new_iter)

with open("src/model_predictive_control/constraints.py", "w") as f:
    f.write(content)
