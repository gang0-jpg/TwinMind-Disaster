import argparse
import subprocess


def run(cmd):
    print(f"[run] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--predict_config", required=True)
    args = parser.parse_args()

    run(f"python scripts/train_unet_cases_unet.py --config {args.train_config}")
    run(f"python scripts/predict_unet_cases_unet.py --config {args.predict_config}")

    print("[done] full pipeline finished")


if __name__ == "__main__":
    main()
