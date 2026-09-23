"""Run a synthetic counterfactual after installing cpmodel-jax."""

from cpmodel_jax import load_example, solve


def main():
    result = solve(load_example("full"))
    print("Wage ratios:", result.values["wage_ratio"])
    print("Welfare ratios:", result.values["welfare_ratio"])
    print("Residual:", result.diagnostics["residual_inf"])
    print("Newton steps:", int(result.root.steps))
    print("Seconds (including first-call compilation):", result.wall_seconds)


if __name__ == "__main__":
    main()
