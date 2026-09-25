import argparse
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat

from cpmodel_jax import solve_eha

DATA_URL = "https://spinup-000d1a-wp-offload-media.s3.amazonaws.com/faculty/wp-content/uploads/sites/40/2019/06/Data_and_Codes_CP.zip"
DATA_SHA256 = "6e68475a047b824890b53f1466c745ac6c5553416a0396b4df1d9c0ce9be6f5a"
NAFTA = {"Mexico": 19, "Canada": 4, "USA": 29}
PUBLISHED_WELFARE = np.array(
    [[1.31, -0.41, 1.72, 1.72], [-0.06, -0.11, 0.04, 0.32], [0.08, 0.04, 0.04, 0.11]]
)


def load_nafta(archive_path):
    archive_path = Path(archive_path)
    download = not archive_path.exists()
    if download:
        print("Downloading the authors' replication archive...")
        with urlopen(DATA_URL, timeout=60) as response:
            content = response.read()
    else:
        content = archive_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != DATA_SHA256:
        raise ValueError("The author archive changed; verify its contents before using it.")
    if download:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(content)
    folder = "Data_and_Codes_CP/Counterfactuals/"
    with ZipFile(io.BytesIO(content)) as archive:
        source = loadmat(io.BytesIO(archive.read(folder + "initial_condition_1993_noS.mat")))
        tariffs_1993 = np.loadtxt(io.BytesIO(archive.read(folder + "tariffs1993.txt")))
        tariffs_2005 = np.loadtxt(io.BytesIO(archive.read(folder + "tariffs2005.txt")))
    countries, sectors = int(source["N"].item()), int(source["J"].item())
    if (countries, sectors) != (31, 40) or np.any(source["Sn"] != 0):
        raise ValueError("Expected the authors' 31-country, 40-sector, no-deficit baseline.")

    def bilateral(array):
        return array.reshape(sectors, countries, countries).transpose(1, 2, 0)

    beta = source["B"].T
    conditional_io_shares = source["G"].reshape(countries, sectors, sectors).transpose(0, 2, 1)
    conditional_io_shares = np.maximum(conditional_io_shares, 0)
    input_share_totals = conditional_io_shares.sum(2, keepdims=True)
    conditional_io_shares = np.divide(
        conditional_io_shares,
        input_share_totals,
        out=np.zeros_like(conditional_io_shares),
        where=input_share_totals > 0,
    )
    gamma = (1 - beta[:, :, None]) * conditional_io_shares

    nontradable_tariffs = np.zeros((20 * countries, countries))
    baseline_tariff_rates = bilateral(np.vstack((tariffs_1993, nontradable_tariffs))) / 100
    tariffs_2005 = bilateral(np.vstack((tariffs_2005, nontradable_tariffs))) / 100
    counterfactual_tariff_rates = baseline_tariff_rates.copy()
    for importer in NAFTA.values():
        for exporter in NAFTA.values():
            if importer != exporter:
                counterfactual_tariff_rates[importer, exporter] = tariffs_2005[importer, exporter]

    return {
        "theta": 1 / source["T"].ravel(),
        "alpha": source["alphas"].T,
        "beta": beta,
        "gamma": gamma,
        "baseline_net_trade_value": bilateral(source["xbilattau"]),
        "baseline_tariff_rates": baseline_tariff_rates,
        "counterfactual_tariff_rates": counterfactual_tariff_rates,
    }


def paper_welfare(inputs, result):
    baseline_net_trade_value = inputs["baseline_net_trade_value"]
    baseline_tariff_rates = inputs["baseline_tariff_rates"]
    baseline_income = (inputs["beta"] * baseline_net_trade_value.sum(0)).sum(1)
    baseline_income += (baseline_net_trade_value * baseline_tariff_rates).sum((1, 2))
    unit_cost_ratio = result["unit_cost_ratio"][None, :, :]
    trade_cost_change = baseline_net_trade_value * (unit_cost_ratio - 1)
    terms_of_trade = (
        trade_cost_change.sum((0, 2)) - trade_cost_change.sum((1, 2))
    ) / baseline_income
    volume_of_trade = (
        baseline_tariff_rates
        * (result["counterfactual_net_trade_value"] - baseline_net_trade_value * unit_cost_ratio)
    ).sum((1, 2)) / baseline_income
    return 100 * np.column_stack(
        (
            terms_of_trade + volume_of_trade,
            terms_of_trade,
            volume_of_trade,
            result["real_wage_ratio"] - 1,
        )
    )


def main():
    parser = argparse.ArgumentParser(description="NAFTA tariff reductions, 1993–2005")
    parser.add_argument(
        "--archive", type=Path, default=Path(__file__).resolve().parents[1] / ".cache/cp2015.zip"
    )
    arguments = parser.parse_args()
    inputs = load_nafta(arguments.archive)
    result = solve_eha(**inputs)
    country_indices = list(NAFTA.values())
    welfare_table = paper_welfare(inputs, result)[country_indices]
    np.testing.assert_allclose(welfare_table, PUBLISHED_WELFARE, atol=0.005, rtol=0)
    print("NAFTA tariffs only: 1993 -> 2005; 31 countries, 40 sectors.")
    print("Technology and iceberg costs are unchanged.")
    print("Data preparation: 1 negative IO entry set to zero; conditional IO shares normalized.")
    print("\nTable 2 decomposition (%) -- NOT 100 * (welfare_ratio - 1):")
    print(f"{'Country':<10} {'Total':>9} {'ToT':>9} {'VoT':>9} {'Real wage':>10}")
    for country, row in zip(NAFTA, welfare_table, strict=True):
        print(f"{country:<10} {row[0]:9.4f} {row[1]:9.4f} {row[2]:9.4f} {row[3]:10.4f}")
    print("All 12 Table 2 entries agree at the published two-decimal precision.")
    print("\nDirect real-income changes (%):", 100 * (result["welfare_ratio"][country_indices] - 1))
    print("JAX dtype:", result["log_ratios"].dtype)
    print("Newton iterations:", result["iterations"])
    print("Maximum equation error:", max(result["diagnostics"].values()))
    print("Solve seconds (includes JIT compilation):", result["wall_seconds"])


if __name__ == "__main__":
    main()
