
#!/usr/bin/env python3
"""Graph tester: runs the fast (patched) generator if requested, then validates
against a baseline directory using graph_validator.py.

Typical usage:
  # 1) You already generated baseline graphs with the original script into BASE_DIR
  # 2) Generate new graphs into NEW_DIR using your patched script (or let this tester run it)
  # 3) Validate:
  python test_graph_parity.py --baseline_dir BASE_DIR --candidate_dir NEW_DIR --report_csv parity_report.csv

Optionally, you can ask this tester to run the generator for you:
  python test_graph_parity.py --gen_cmd "python make_graph_db_fast_patched.py --dataset DRIVE --source_type result --edge_method geo_dist --edge_dist_thresh 10 --win_size 4" --candidate_dir NEW_DIR --baseline_dir BASE_DIR

Notes:
- This script does not assume a particular dataset layout; it just shells out to your generator command if provided.
- Validation thresholds can be passed through to the validator via flags here.
"""
import argparse
import os
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser(description='Run generator (optional) and validate graph parity.')
    ap.add_argument('--gen_cmd', type=str, default=None, help='Optional shell command to generate candidate graphs before validation.')
    ap.add_argument('--baseline_dir', type=str, required=True, help='Directory containing baseline .graph_res files.')
    ap.add_argument('--candidate_dir', type=str, required=True, help='Directory containing candidate .graph_res files.')
    ap.add_argument('--pixel_tol_soft', type=int, default=1)
    ap.add_argument('--edge_thresh', type=float, default=0.98)
    ap.add_argument('--node_thresh', type=float, default=0.99)
    ap.add_argument('--deg_l1_thresh', type=float, default=0.05)
    ap.add_argument('--report_csv', type=str, default='parity_report.csv')
    args = ap.parse_args()

    if args.gen_cmd:
        print('Running generator command...')
        print(args.gen_cmd)
        ret = subprocess.run(args.gen_cmd, shell=True)
        if ret.returncode != 0:
            print('Generator command failed with non-zero exit code:', ret.returncode)
            sys.exit(ret.returncode)

    # Build validator command
    validator = os.path.join(os.path.dirname(__file__), 'graph_validator.py')
    cmd = [
        sys.executable, validator,
        '--baseline_dir', args.baseline_dir,
        '--candidate_dir', args.candidate_dir,
        '--pixel_tol_soft', str(args.pixel_tol_soft),
        '--edge_thresh', str(args.edge_thresh),
        '--node_thresh', str(args.node_thresh),
        '--deg_l1_thresh', str(args.deg_l1_thresh),
        '--report_csv', args.report_csv
    ]
    print('Validating parity...')
    print(' '.join(cmd))
    ret = subprocess.run(cmd)
    sys.exit(ret.returncode)

if __name__ == '__main__':
    main()
