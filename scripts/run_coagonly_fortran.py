"""Run Fortran sulfatetest_ensemble with H2SO4=0 to get coag-only behavior."""
import numpy as np, json, subprocess, tempfile, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

BINARY = os.path.expanduser('~/Documents/GitHub/aerosol-microphysics/carma-app/original-carma/CARMA/build/test_sulfate_ensemble')

def run_one(idx, scen):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        # Force H2SO4 = 0 → no growth/nucleation, only coag operates
        f.write(f'{scen["T"]:.4f}  {scen["p"]:.4f}  {scen["rh"]:.4f}  0.0  {scen["aerosol_mu_nm"]:.4f}  {scen["aerosol_sigma_g"]:.4f}\n')
        scen_path = f.name
    out_path = scen_path + '.out'
    r = subprocess.run([BINARY, scen_path, out_path], capture_output=True, text=True, timeout=300)
    os.unlink(scen_path)
    if r.returncode != 0:
        os.unlink(out_path) if os.path.exists(out_path) else None
        return idx, None
    with open(out_path) as f:
        j = json.loads(f.read())
    os.unlink(out_path)
    return idx, np.array(j['pc_final'])

def main():
    sc = np.load('data/sulfate_scenarios_realistic_1000.npz')
    n = 1000
    scens = [{k: float(sc[k][i]) for k in ['T','p','rh','aerosol_mu_nm','aerosol_sigma_g']} for i in range(n)]
    pc_all = np.zeros((n, 38), dtype=np.float64)
    print(f'Running coag-only Fortran on {n} scenarios (H2SO4=0)...')
    with ProcessPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(run_one, i, scens[i]): i for i in range(n)}
        done = 0
        for fut in as_completed(futures):
            idx, pc = fut.result()
            if pc is not None:
                pc_all[idx] = pc
            done += 1
            if done % 100 == 0:
                print(f'  {done}/{n}', flush=True)
    np.savez_compressed('data/sulfate_fortran_coagonly.npz', pc_final=pc_all)
    print(f'Saved data/sulfate_fortran_coagonly.npz')

if __name__ == '__main__':
    main()
