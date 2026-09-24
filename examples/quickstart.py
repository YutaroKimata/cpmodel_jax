from cpmodel_jax import load_example, solve

inputs = load_example("full")
result = solve(**inputs, tolerance=1e-10)
print("Welfare percentage changes:", 100 * (result["welfare_ratio"] - 1))
print("Newton iterations:", result["iterations"])
print("Equation errors:", result["diagnostics"])
